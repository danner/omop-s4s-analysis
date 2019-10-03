import os
import glob
import csv
import pprint as pp
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from functools import reduce

import omop_analyze
import fhir_analyze

# Helper functions:
def configure_tables():
    pd.set_option('max_rows', 999)
    pd.set_option('max_colwidth', 120)

def path_for_resource(resource):
    resource_type = resource['resourceType']
    code_paths = {
        #'OperationOutcome': ['details', 'coding'],
        'OperationOutcome': ['issue', 'details', 'coding'],
        'MedicationOrder': ['medicationCodeableConcept', 'coding'], # no idea what I actually need here
        'MedicationStatement': ['medicationCodeableConcept', 'coding'],
        'AllergyIntolerance': ['substance', 'coding'],
        'Observation': ['category', 'coding'], #code, coding
        'Immunization': ['vaccineCode', 'coding'],
        'Condition': ['category', 'coding'], #code, coding
        'DocumentReference': ['class', 'coding'],
        'Procedure': ['code', 'coding'],

    }
    return code_paths[resource_type]

def fetch_at_path(resource, path):
    if type(path) == type(''):
        path = path.split('.')
    def walk(data, k):
        if isinstance(data, dict):
            return data.get(k)
        elif isinstance(data, list):
            return [reduce(walk, [k], el) for el in data]
        return None
    return reduce(walk, path, resource)

class Node:
    def __init__(self, parent=None):
        self.count = Counter()
        self.children = {}
        self.parent_node = parent
        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def print_children(self, children):
        return "".join([
            " Children: {{{}".format("\n" if children else ""),
            "".join("{}{}: {}\n".format("   "*self.depth, k, v) for k, v in children.items()),
            "{}}}".format("   "*(self.depth-1) if children else ""),
        ])

    def __repr__(self):
        children = self.children
        return "Node: {count}{children}".format(**{
            "count": dict(self.count),
            "children": self.print_children(children),
        })

    def __str__(self):
        children = self.children
        return "Node: {count}{children}".format(**{
            "count": dict(self.count),
            "children": self.print_children(children),
        })

def traverse(resource, node):
    node.count[type(resource)] += 1
    if isinstance(resource, dict):
        for k, v in resource.items():
            if k not in node.children:
                node.children[k] = Node(parent=node)
            traverse(v, node.children[k])
    elif isinstance(resource, list):
        for i, item in enumerate(resource):
            if i not in node.children:
                node.children[i] = Node(parent=node)
            traverse(item, node.children[i])
    return node

# Report Functions
def fhir_plot_category_counts(fhir_people):
    s4s_datatype_totals = {person:{title:len(items) for (title, items) in datatype.items()} for (person, datatype) in fhir_people.items()}
    s4s_df = pd.DataFrame(s4s_datatype_totals).transpose()
    return s4s_df.plot(kind='box', figsize=(20,6), logy=True)

def omop_plot_category_counts(omop_people, categories):
    omop_data_types_per_person = {
        data_type: [
            len(person[data_type]) if data_type in person.keys() else 0 for person in omop_people.values()
        ] for data_type in categories
    }
    omop_df_types = pd.DataFrame(omop_data_types_per_person, index=omop_people.values())
    return omop_df_types.boxplot(figsize=(12,6), showfliers=False)

def code_system_counts(fhir_people):
    # Count of code *systems* for each data category. E.g., fraction of SNOMED vs LOINC vs Other codes found in Conditions.
    coding_paths = {}
    for person, documents in fhir_people.items():
        for document, data in documents.items():
            if document not in coding_paths:
                coding_paths[document] = Counter()
            for entry in data:
                fetched = fetch_at_path(entry, path_for_resource(entry))
                if fetched:
                    for f in fetched:
                        try:
                            coding_paths[document][f.get('system')] += 1
                        except AttributeError:
                            pass
    return coding_paths

def coding_counts(fhir_people):
    # Count of codings for each data category.
    coding_paths = {}
    for person, documents in fhir_people.items():
        for document, data in documents.items():
            if document not in coding_paths:
                coding_paths[document] = Counter()
            for entry in data:
                fetched = fetch_at_path(entry, path_for_resource(entry))
                if fetched:
                    for f in fetched:
                        try:
                            coding_paths[document][f.get('system', 'None')+' '+f.get('code', 'None')+' '+f.get('display', 'None')] += 1
                        except AttributeError:
                            pass
    return coding_paths

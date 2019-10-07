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
    pd.set_option('display.width', 150)

def path_for_resource(resource):
    resource_type = resource['resourceType']
    code_paths = {
        #'OperationOutcome': ['details', 'coding'],
        'OperationOutcome': ['issue', 'details', 'coding'],
        'MedicationOrder': ['medicationCodeableConcept', 'coding'], # no idea what I actually need here
        'MedicationStatement': ['medicationCodeableConcept', 'coding'],
        'AllergyIntolerance': ['substance', 'coding'],
        'Observation': ['code', 'coding'], #code, coding
        'Immunization': ['vaccineCode', 'coding'],
        'Condition': ['code', 'coding'], #code, coding
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

CODE_COLUMNS = {
    'condition.csv': ('condition_concept_id', 'condition_source_concept_id'),
    'observation.csv': ('observation_concept_id', 'observation_source_concept_id'),
    'procedure.csv': ('procedure_concept_id', 'procedure_source_concept_id'),
    'drug_summary.csv': ('drug_concept_id', 'drug_source_concept_id'),
    'drug.csv': ('drug_concept_id', 'drug_source_concept_id'),
    'measurement.csv': ('measurement_concept_id', 'measurement_source_concept_id'),
}

def csv_to_dicts(filename):
    with open(filename, encoding="utf8") as csv_file:
        items = (filename, list(csv.DictReader(csv_file, delimiter="\t")))
    return items

def init_omop_concepts():
    vocab = csv_to_dicts('VOCABULARY.csv')[1]
    vocab_df = pd.DataFrame(vocab)
    concept = csv_to_dicts('CONCEPT.csv')[1]
    concept_df = pd.DataFrame(concept)
    concept_cpt4 = csv_to_dicts('CONCEPT_CPT4.csv')[1]
    cpt4_df = pd.DataFrame(concept_cpt4)
    concept_aouppi = csv_to_dicts('CONCEPT_AOUPPI.csv')[1]
    aouppi_df = pd.DataFrame(concept_aouppi)
    concept_df.set_index(['concept_id',], inplace=True)
    cpt4_df.set_index(['concept_id',], inplace=True)
    aouppi_df.set_index(['concept_id',], inplace=True)
    vocab_df.set_index(['vocabulary_id',], inplace=True)

    CONCEPT_TABLES = [concept_df, cpt4_df, aouppi_df]
    return CONCEPT_TABLES

#quick and dirty caching
def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper

missing_concept_codes = set()

@memoize
def omop_concept_lookup(concept_id):
    concept = None
    for table in CONCEPT_TABLES:
        try:
            concept = table.loc[concept_id]
            if not concept.empty:
                break
        except KeyError:
            continue
    return concept

def omop_source_concept_code(concept_id):
    concept = omop_concept_lookup(concept_id)
    try:
        return concept.concept_code
    except AttributeError:
        missing_concept_codes.add(concept_id)
        return None

def omop_concept_vocabulary_id(concept_id):
    concept = omop_concept_lookup(concept_id)
    try:
        return concept.vocabulary_id
    except AttributeError:
        return None

def omop_concept_name(concept_id):
    concept = omop_concept_lookup(concept_id)
    try:
        return concept.concept_name
    except AttributeError:
        return None

def convert_vocabulary(system):
    converter = {
        'http://loinc.org': 'LOINC',
        'http://snomed.info/sct': 'SNOMED',
        'http://hl7.org/fhir/sid/icd-9-cm/diagnosis': 'ICD9CM',
        'http://www.ama-assn.org/go/cpt': 'CPT4',
        'http://hl7.org/fhir/sid/icd-9-cm': 'ICD9CM',
        'http://hl7.org/fhir/sid/icd-10-cm': 'ICD10CM',
        'urn:oid:2.16.840.1.113883.6.90': 'ICD10CM',
        'urn:oid:2.16.840.1.113883.6.14': 'HCPCS',
        'http://www.nlm.nih.gov/research/umls/rxnorm': 'RxNorm',
        'http://hl7.org/fhir/sid/ndc': 'NDC',
        'http://hl7.org/fhir/ndfrt': 'None',
        'http://fdasis.nlm.nih.gov': 'None',
        'http://hl7.org/fhir/sid/cvx': 'CVX',
        'http://hl7.org/fhir/observation-category': 'Observation Type',
        'https://apis.followmyhealth.com/fhir/id/translation': 'None',
        'http://argonautwiki.hl7.org/extension-codes': 'None',
        'http://hl7.org/fhir/condition-category': 'None',
        'http://argonaut.hl7.org': 'None',
        # these are codes for EPIC clients. I need to figure out what the suffix means.
        'urn:oid:1.2.840.114350.1.13.362.2.7.2.696580': 'None',
        'urn:oid:1.2.840.114350.1.13.202.2.7.2.696580': 'None',
        'urn:oid:1.2.840.114350.1.13.71.2.7.2.696580': 'None',
        'urn:oid:1.2.840.114350.1.13.324.2.7.2.696580': 'None',
    }
    try:
        return converter[system]
    except KeyError:
        print(system)
        return system

def omop_concept_to_coding(row, table):
    return tuple((omop_concept_vocabulary_id(row[column]), omop_source_concept_code(row[column])) for column in CODE_COLUMNS[table] if column in row.keys())


CONCEPT_TABLES = init_omop_concepts()

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
    display_codes = {}
    for person, documents in fhir_people.items():
        for document, data in documents.items():
            if document not in coding_paths:
                coding_paths[document] = Counter()
            for entry in data:
                fetched = fetch_at_path(entry, path_for_resource(entry))
                if fetched:
                    for f in fetched:
                        try:
                            coding_paths[document][f.get('system', 'None')+' '+f.get('code', 'None')] += 1
                            display_codes[f.get('system', 'None')+' '+f.get('code', 'None')] = {
                                'system': f.get('system', 'None'),
                                'code': f.get('code', 'None'),
                                'display': f.get('display', 'None')
                            }
                        except AttributeError:
                            pass
    #now make a table with display values
    coding_table = {}
    for category, counter in coding_paths.items():
        coding_table[category] = [{**display_codes[coding], **{'count': count}} for coding, count in counter.most_common()]
    return coding_table

def omop_system_counts(omop_people):
    # Count of standardized code *systems* for each OMOP data type. E.g., fraction of SNOMED vs LOINC vs Other codes found in condition_concept_id.
    systems = Counter()
    for person, tables in omop_people.items():
        for filename, incidents in tables.items():
            for incident in incidents:
                coding = omop_concept_to_coding(incident, filename)
                if not coding[0][0] or coding[0][0] == 'None':
                    systems['None'] += 1
                else:
                    systems[coding[0][0]] += 1

    return systems

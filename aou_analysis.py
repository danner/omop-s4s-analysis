import os
import glob
import csv
import uuid
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
        self.type = None
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
        if self.type is type([]):
            return "{type} node: top values: {count}{children}".format(**{
                "type": self.type,
                "count": self.count.most_common(5),
                "children": self.print_children(children),
            })
        elif self.type is type({}):
            return "{type} node: {children}".format(**{
                "type": self.type,
                "children": self.print_children(children),
            })
        elif self.type is type(1) or type("str"):
            return "{type} node: top values: {count}".format(**{
                "type": self.type,
                "count": self.count.most_common(),
            })
        else:
            return "*****************{type} node: {count}{children}".format(**{
                "type": self.type,
                "self": self.__dict__,
            })

    def __str__(self):
        children = self.children
        return "Node: {count}{children}".format(**{
            "type": self.type,
            "count": self.count.most_common(5),
            "children": self.print_children(children),
        })

def traverse(resource, node):
    node.type = type(resource).__name__
    if isinstance(resource, dict):
        for k, v in resource.items():
            if k not in node.children:
                node.children[k] = Node(parent=node)
            traverse(v, node.children[k])
    elif isinstance(resource, list):
        self.count[len(resource)] += 1
        for item in resource:
            if len(resource) not in node.children:
                node.children[len(resource)] = Node(parent=node)
            traverse(item, node.children[i])
    else:
        self.count[resource] += 1
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

def most_common_synonym(coding_sets):
    coding2hash = {}
    hash2set = {}
    most_common = Counter()
    for i, coding_set in enumerate(coding_sets):
        print("set", i, "of ", len(coding_sets))
        for coding in coding_set:
            most_common[coding] += 1
        if len(coding_set) < 2:
            continue
        current_hash = None
        for coding in coding_set:
            #first off, we're looking if any of these codings already exist.
            try:
                current_hash = coding2hash[coding]
                break
            except KeyError:
                # this coding doesn't exist as a synonym yet.
                # on to the next.
                print(coding, "wasn't in coding2hash")
                continue
        #once we're done looking at all the codings in the coding set
        #decide what to do.
        if current_hash:
            combine_these = set()
            hash2set[current_hash].update(coding_set)
            for coding in coding_set:
                try:
                    if coding2hash[coding] != current_hash:
                        # this coding belongs to a different set.
                        # add that set to the current set
                        combine_these.update(hash2set[coding2hash[coding]])
                except KeyError:
                    # why doesn't this coding exist in coding2hash?
                    print(coding, "wasn't found in coding2hash?")
            if len(combine_these):
                # to combine, we grab each hash from codings,
                hashes = set()
                for coding in combine_these:
                    hashes.add(coding2hash[coding])
                    # update the coding2hash,
                    coding2hash[coding] = current_hash
                # then run through the hashes, add them together
                for synonym_hash in hashes:
                    # delete them from the sets, and append them to the current_hash
                    try:
                        hash2set[current_hash].update(hash2set.pop(synonym_hash))
                    except KeyError as e:
                        print(combine_these, hash2set, e)
        else:
            #new set! easy peasy.
            new_hash = uuid.uuid4()
            hash2set[new_hash] = coding_set
            coding2hash.update({coding:new_hash for coding in coding_set})
    # all synonyms are now combined.
    print("now generating mapping between coding and most common synonym")
    most_common_synonym = {}
    for synonym_set in hash2set.values():
        if len(synonym_set):
            most_seen = max(list(synonym_set), key=lambda synonym:  most_common[synonym])
            for synonym in synonym_set:
                most_common_synonym[synonym] = most_seen
        else:
            print("empty synonym set++")

    return most_common_synonym

CONCEPT_TABLES = init_omop_concepts()

# Report Functions - FHIR
def fhir_plot_category_counts(fhir_people):
    s4s_datatype_totals = {person:{title:len(items) for (title, items) in datatype.items()} for (person, datatype) in fhir_people.items()}
    s4s_df = pd.DataFrame(s4s_datatype_totals).transpose()
    return s4s_df.plot(kind='box', figsize=(20,6), logy=True)

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
    coding_sets = {}
    display_codes = {}
    for person, documents in list(fhir_people.items())[:1]:
        for document, data in documents.items():
            if document not in coding_paths:
                coding_paths[document] = Counter()
                coding_sets[document] = []
            for entry in data:
                fetched = fetch_at_path(entry, path_for_resource(entry))
                if fetched:
                    coding_set = set()
                    for f in fetched:
                        try:
                            coding = {
                                'system': f.get('system', 'None'),
                                'code': f.get('code', 'None'),
                                'display': f.get('display', 'None')
                            }
                        except AttributeError:
                            #for some reason there are codings that are lists
                            f = f[0]
                            coding = {
                                'system': f.get('system', 'None'),
                                'code': f.get('code', 'None'),
                                'display': f.get('display', 'None')
                            }
                        code_hash = coding['system']+' '+coding['code']
                        if f == fetched[0]:
                            #maybe I should only add the first one, because we're using synonyms.
                            coding_paths[document][code_hash] += 1
                        coding_set.add(code_hash)
                        display_codes[code_hash] = coding
                    coding_sets[document].append(coding_set)
    #work out the most common synonyms
    most_common_coding = {}
    for document in coding_sets.keys():
        most_common_coding[document] = most_common_synonym(coding_sets[document])
    #combine the counts of synonyms
    common_counter = {}
    for category, counter in coding_paths.items():
        if category not in common_counter:
            common_counter[category] = Counter()
        for coding, count in counter.most_common():
            common_counter[category][most_common_coding[category][coding]] += count

    #now make a table with display values
    coding_table = {}
    for category, counter in common_counter.items():
        coding_table[category] = [{**display_codes[coding], **{'count': count}} for coding, count in counter.most_common()]
    return coding_table

# Report Functions - OMOP

def omop_plot_category_counts(omop_people, categories):
    omop_data_types_per_person = {
        data_type: [
            len(person[data_type]) if data_type in person.keys() else 0 for person in omop_people.values()
        ] for data_type in categories
    }
    omop_df_types = pd.DataFrame(omop_data_types_per_person, index=omop_people.values())
    return omop_df_types.boxplot(figsize=(12,6), showfliers=False)

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

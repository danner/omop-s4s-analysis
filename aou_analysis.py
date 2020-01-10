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

def merge_multi(self, df, on):
    return self.reset_index().join(df,on=on).set_index(self.index.names)
pd.DataFrame.merge_multi = merge_multi

def export_df(df, filename):
    df.to_csv(path_or_buf=filename)

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
        'Patient': ['code', 'coding'],
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

STATUS_WHITELIST = [
    'status',
    'system',
    'clinicalStatus',
    'verificationStatus',
    'resourceType',
]
class Node:
    def __init__(self, parent=None):
        self.type = None
        self.name = None
        self.count = Counter()
        self.children = {}
        self.parent_node = parent
        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def print_children(self):
        return "".join([
            " children: {{{}".format("\n" if self.children else ""),
            "".join("{}{}: {}\n".format("   "*(self.depth+1), k, v) for k, v in self.children.items()),
            "{}}}".format("   "*(self.depth) if self.children else ""),
        ])
    def full_path(self):
        names = []
        current_node = self
        while current_node.parent_node:
            if current_node.name:
                names.append(current_node.name)
            current_node = current_node.parent_node
        return ".".join(reversed(names))

    def convert_to_dict(self):
        conversion = {}
        conversion['type'] = self.type
        conversion['name'] = self.name
        conversion['depth'] = self.depth
        if self.type == "list":
            conversion['list_lengths'] = self.count.most_common()
        elif self.type in ["str", "int", "float", "bool"]:
            conversion['count'] = sum(self.count.values())
            if any([self.name in STATUS_WHITELIST,
                    self.type == 'bool',
                    self.name == "url" and "extension.url" in self.full_path(),
                ]):
                conversion['top_values'] = {}
                for k, v in self.count.most_common():
                    conversion['top_values'][k] = v
        if self.children:
            conversion['children'] = []
            for k in self.children:
                conversion['children'].append(self.children[k].convert_to_dict())
        return conversion

    def __repr__(self):
        if self.type == "list":
            return "<{type} top values: {count}{children}>".format(**{
                "type": self.type,
                "count": self.count.most_common(5),
                "children": self.print_children(),
            })
        elif self.type == "dict":
            return "<{type}{children}>".format(**{
                "type": self.type,
                "children": self.print_children(),
            })
        elif self.type in ["str", "int", "float", "bool"]:
            count = str(sum(self.count.values()))
            if any([self.name in STATUS_WHITELIST,
                    self.type == 'bool',
                    self.name == "url" and "extension.url" in self.full_path(),
                ]):
                count += " values: {statuses}".format(**{
                    "statuses": "".join("\n{}{}: {}".format(
                        "   "*(self.depth+1), k, v
                    ) for k, v in self.count.most_common()),
                })
            return "<{type} total: {count}>".format(**{
                "type": self.type,
                "count": count,
            })
        else:
            return "<{type} node  needs investigation>".format(**{
                "type": self.type,
            })

    def __str__(self):
        return self.__repr__()

def traverse(resource, node, name=None):
    node.type = type(resource).__name__
    node.name = name
    if isinstance(resource, dict):
        for k, v in resource.items():
            if k not in node.children:
                node.children[k] = Node(parent=node)
            traverse(v, node.children[k], k)
    elif isinstance(resource, list):
        node.count[str(len(resource))] += 1
        for item in resource:
            if type(item).__name__ not in node.children:
                node.children[type(item).__name__] = Node(parent=node)
            traverse(item, node.children[type(item).__name__])
    else:
        node.count[resource] += 1
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

    vocab_df.set_index(['vocabulary_id',], inplace=True)

    CONCEPT_TABLES = [concept_df, cpt4_df, aouppi_df]
    merged_concept_df = concept_df.append([cpt4_df, aouppi_df])
    merged_concept_df.set_index(['concept_id', 'concept_code', 'vocabulary_id',], inplace=True)
    return merged_concept_df

NO_DATA = 'Empty raw value'
MISSING_CONCEPT = 'Missing concept'
NO_MATCHING_CONCEPT = 'No standardized concept'
NO_MATCHING_DISPLAY = 'No standardized display'
PARSE_ERROR = 'Error parsing'

missing_concept_codes = set()

def get_fhir_standardized_concept(fhir_coding):
    standardized_concept = concept_table.iloc[
        concept_table.index.isin([fhir_coding['code']], level=1) &
        concept_table.index.isin([
            convert_vocabulary(fhir_coding['system'])],
            level=2
        )]
    if len(standardized_concept) < 1:
        return NO_MATCHING_CONCEPT
    else:
        return standardized_concept

def get_fhir_standardized_concept_name(fhir_coding):
    try:
        concept = get_fhir_standardized_concept(fhir_coding)
        return concept['concept_name'].values[0]
    except IndexError:
        return NO_MATCHING_DISPLAY
    except TypeError:
        if concept == NO_MATCHING_CONCEPT:
            return NO_MATCHING_CONCEPT


class Memoize:
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.fn(*args)
        return self.memo[args]

@Memoize
def omop_concept_lookup(concept_id):
    concept = concept_id.split('.')[0]
    if not concept:
        return MISSING_CONCEPT
    try:
        return concept_table.loc[concept]
    except KeyError as e:
        #print("couldn't find concept id:", concept)
        missing_concept_codes.add(concept_id)
        return NO_MATCHING_CONCEPT
    except ValueError as e:
        print("what's this concept?", concept_id.__dict__)
        return PARSE_ERROR

def omop_source_concept_code(concept_id):
    concept = omop_concept_lookup(concept_id)
    try:
        return concept.index.values[0][0]
    except AttributeError:
        return NO_MATCHING_CONCEPT

def omop_concept_vocabulary_id(concept_id):
    concept = omop_concept_lookup(concept_id)
    try:
        return concept.index.values[0][1]
    except AttributeError:
        return NO_MATCHING_CONCEPT

def omop_concept_name(concept_id):
    concept = omop_concept_lookup(concept_id)
    try:
        return concept.concept_name
    except AttributeError:
        return NO_MATCHING_CONCEPT

def concept_code_query(df, system, code):
    # need to find a better way to query this. maybe a join?
    concept = None
    try:
        concept = df.query(
            "vocabulary_id == '{}' and concept_code == '{}'".format(
                system,
                code,
            )
        )
    except KeyError as e:
        print(e)
    return concept

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
        print("found a missing system:", system)
        return system

def omop_raw_coding(row, table):
    concepts = []
    for column in CODE_COLUMNS[table]:
        if column in row.keys():
            concepts.append(row[column])
        else:
            concepts.append('None')
    return " ".join(concepts)

def omop_concept_to_coding(row, table):
    concepts = []
    for column in CODE_COLUMNS[table]:
        try:
            concepts.append(omop_concept_lookup(row[column]))
        except AttributeError:
            concepts.append(NO_MATCHING_CONCEPT)
    try:
        return tuple({
            'system': concept.index.values[0][1],
            'coding': concept.index.values[0][0],
            'name': concept.concept_name.values[0],
        } for concept in concepts)
    except AttributeError as e:
        #print("no concept for ", row, table)
        return ({}, {})

def compose_vocab_df(vocab):
    vocab_df = pd.DataFrame(vocab).transpose()
    vocab2_df = vocab_df.join(vocab_df[0].apply(pd.Series)['name'])
    vocab2_df.rename(columns={'name': 'concept_name'}, inplace=True)
    vocab3_df = vocab2_df.join(vocab_df[1].apply(pd.Series)['name'])
    vocab3_df.rename(columns={'name': 'source_name'}, inplace=True)
    vocab4_df = vocab3_df.drop(columns=[0,1])
    return vocab4_df

def most_common_synonym(coding_sets):
    coding2hash = {}
    hash2set = {}
    most_common = Counter()
    for i, coding_set in enumerate(coding_sets):
        #print("set", i, "of ", len(coding_sets), "with a length of", len(coding_set))
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
                continue
        #once we're done looking at all the codings in the coding set
        #decide what to do.
        if current_hash:
            combine_these = set()
            #prep the cascading synonym collection
            for coding in coding_set:
                target_hash = coding2hash.get(coding, None)
                if target_hash != current_hash:
                    # this coding belongs to a different set.
                    # prep to add that set to the current set
                    #print("found a coding", coding,
                    #      "in another hash", coding2hash.get(coding, None))
                    combine_these.add(coding)
            while len(combine_these):
                #print(len(combine_these), "left to combine")
                current_coding = combine_these.pop()
                target_hash = coding2hash.get(current_coding, None)
                if target_hash != current_hash and target_hash:
                    combine_these.update(hash2set.get(target_hash, []))
                    #just make sure we aren't this again.
                    combine_these.discard(current_coding)
                # update the coding2hash and hash2set
                coding2hash[current_coding] = current_hash
                hash2set[current_hash].add(current_coding)
        else:
            #new set! easy peasy.
            new_hash = uuid.uuid4()
            #print("new synonym set found", new_hash)
            hash2set[new_hash] = coding_set
            coding2hash.update({coding:new_hash for coding in coding_set})
    # all synonyms are now combined.
    #print("now generating mapping between coding and most common synonym")
    most_common_synonym = {}
    for synonym_set in hash2set.values():
        if len(synonym_set):
            most_seen = max(list(synonym_set), key=lambda synonym:  most_common[synonym])
            for synonym in synonym_set:
                most_common_synonym[synonym] = most_seen

    #print("made a synonym list of length ", len(most_common_synonym.keys()))
    return most_common_synonym

concept_table = init_omop_concepts()

# Report Functions - FHIR
def fhir_plot_category_counts(fhir_people):
    s4s_datatype_totals = {person:{title:len(items) for (title, items) in datatype.items()} for (person, datatype) in fhir_people.items()}
    s4s_df = pd.DataFrame(s4s_datatype_totals).transpose()
    return s4s_df

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
    for person, documents in list(fhir_people.items()):
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
                                'system': f.get('system', NO_DATA),
                                'code': f.get('code', NO_DATA),
                                'display': f.get('display', NO_DATA)
                            }
                        except AttributeError:
                            #for some reason there are codings that are lists
                            f = f[0]
                            coding = {
                                'system': f.get('system', NO_DATA),
                                'code': f.get('code', NO_DATA),
                                'display': f.get('display', NO_DATA)
                            }
#                         if coding['display'] == NO_DATA:
#                             coding['display'] = get_fhir_standardized_concept_name(coding)
                        code_hash = coding['system']+' '+coding['code']
                        if f == fetched[0]:
                            #maybe I should only add the first one, because we're using synonyms.
                            coding_paths[document][code_hash] += 1
                        coding_set.add(code_hash)
                        display_codes[code_hash] = coding
                    coding_sets[document].append(coding_set)
    #work out the most common synonyms
    most_common_coding = {}
    synonym_sets = {}
    for document in coding_sets.keys():
        most_common_coding[document] = most_common_synonym(coding_sets[document])
        synonym_sets[document] = {}
        # once we've got a lookup dict, also create a mapping from
        # the most common synonym to all its less popular codings.
        for key, value in most_common_coding[document].items():
            if value in synonym_sets[document]:
                synonym_sets[document][value].append(key)
            else:
                synonym_sets[document][value]=[key]

    #combine the counts of synonyms
    common_counter = {}
    for category, counter in coding_paths.items():
        if category not in common_counter:
            common_counter[category] = Counter()
        for coding, count in counter.most_common():
            try:
                common_counter[category][most_common_coding[category][coding]] += count
            except KeyError:
                #if it's not in most_common_coding, it doesn't have any synonyms.
                #just apply it directly to the common counter.
                common_counter[category][coding] = count

    #now make a table with display values
    coding_table = {}
    for category, counter in common_counter.items():
        coding_table[category] = [{**display_codes[coding], **{'count': count}} for coding, count in counter.most_common()]
    return {
        'table': coding_table,
        'synonyms': synonym_sets,
        'display': display_codes,
    }

def print_synonym_sets(synonyms, display_names):
    for key, value in synonyms.items():
        most_common = display_names[key]['display']
        if most_common == 'None':
            most_common = key
        print(most_common, '=>', [display_names[v]['display'] for v in value])


# Report Functions - OMOP

def omop_plot_category_counts(omop_people, categories):
    omop_data_types_per_person = {
        data_type: [
            len(person[data_type]) if data_type in person.keys() else 0 for person in omop_people.values()
        ] for data_type in categories
    }
    omop_df_types = pd.DataFrame(omop_data_types_per_person, index=omop_people.values())
    return omop_df_types

def omop_system_counts(omop_people):
    # Count of standardized code *systems* for each OMOP data type. E.g., fraction of SNOMED vs LOINC vs Other codes found in condition_concept_id.
    systems = {}
    for person, tables in omop_people.items():
        for filename, incidents in tables.items():
            if not filename in systems:
                systems[filename] = Counter()
            for incident in incidents:
                coding = list(omop_concept_to_coding(incident, filename))
                try:
                    systems[filename][coding[0]['system']] += 1
                except KeyError:
                    systems[filename]['None'] += 1
    return systems

def omop_coding_counts(omop_people):
    codes = {}
    standardized_codings = {}
    for person, tables in omop_people.items():
        for filename, incidents in tables.items():
            if filename not in codes:
                codes[filename] = Counter()
            for incident in incidents:
                coding = omop_raw_coding(incident, filename)
                standardized_codings[coding] = list(omop_concept_to_coding(incident, filename))
                codes[filename][coding] += 1
    return codes, standardized_codings

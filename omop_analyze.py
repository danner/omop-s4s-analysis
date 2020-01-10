import csv
import os
import glob
import json
import logging
import argparse

code_column = {
    'condition.csv': 'condition_concept_id',
    'observation_1.csv': 'observation_concept_id',
    'observation_2.csv': 'observation_concept_id',
    'observation.csv': 'observation_concept_id',
    'procedure.csv': 'procedure_concept_id',
    'drug_summary.csv': 'drug_concept_id',
    'drug.csv': 'drug_concept_id',
    'measurement.csv': 'measurement_concept_id',
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='Directory containing omop csv files',
        default='.\\omop\\20190326',
    )
    parser.add_argument(
        '-d',
        '--debug',
        help='Show debug messages',
        action='store_const',
        dest='log_level',
        const=logging.DEBUG,
        default=logging.WARNING,
    )

    return parser.parse_args()

def csv_to_dicts(filename):
    with open(filename, encoding="utf8") as csv_file:
        items = (filename, list(csv.DictReader(csv_file)))
    return items

def ids_for_column(data, column_name):
    return list(set(item[column_name] for item in data))

def id_sets_for_interesting_columns(csvs):
    ids = []
    for filename, table in [csv_to_dicts(csv) for csv in csvs]:
        ids.append({
            code_column[filename]: ids_for_column(table, code_column[filename])
        })
    return ids

def data_dump(path=".\\omop\\20190326", extension='csv'):
    cwd = os.getcwd()
    os.chdir(path)
    csvs = [i for i in glob.glob('*.{}'.format(extension))]
    print(csvs)
    data = []
    for file in csvs:
        dicts = csv_to_dicts(file)
        data.append(dicts)
    os.chdir(cwd)
    return data

def parse_omop(path=".\\omop\\20190326", extension='csv'):
    cwd = os.getcwd()
    os.chdir(path)
    csvs = [i for i in glob.glob('*.{}'.format(extension))]
    # print(csvs)
    patients = {}
    for filename, table in [csv_to_dicts(csv) for csv in csvs]:
        for interaction in table:
            if interaction.get('person_id', False) in patients:
                if filename in patients[interaction['person_id']]:
                    patients[interaction['person_id']][filename].append(interaction)
                else:
                    patients[interaction['person_id']][filename] = [interaction, ]
            else:
                if 'person_id' in interaction:
                    patients[interaction['person_id']] = {filename:[interaction, ]}
                else:
                    print("found line without patient")
    print("Got {} omop participants".format(len(patients.keys())))
    os.chdir(cwd)
    return patients, csvs

def main():
    """Find OMOP csvs and output json
    """
    args = parse_arguments()
    logging.basicConfig(level=args.log_level)
    extension = 'csv'
    return parse_omop(args.path, extension)


if __name__ == '__main__':
    print(json.dumps(main(), indent=2, sort_keys=True))

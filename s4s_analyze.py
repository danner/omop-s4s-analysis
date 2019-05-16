from __future__ import division

import argparse
from collections import Counter, defaultdict
import glob
import json
import logging
import os
import re


FILE_TYPE_MAPPING = {
    'ALLERGY_INTOLERANCE': 'AllergyIntolerance',
    'DOCUMENT': 'DocumentReference',
    'IMMUNIZATION': 'Immunization',
    'LAB': 'Observation',
    'MEDICATION_ADMINISTRATION': 'MedicationAdministration',
    'MEDICATION_DISPENSE': 'MedicationDispense',
    'MEDICATION_ORDER': 'MedicationOrder',
    'MEDICATION_STATEMENT': 'MedicationStatement',
    'PATIENT_DEMOGRAPHICS': 'Patient',
    'PROBLEMS': 'Condition',
    'PROCEDURE': 'Procedure',
    'SMOKING_STATUS': 'Observation',
    'VITAL': 'Observation'
}


BASE_URI_TEMPLATE = r'(?P<base_uri>.*/){}(/[A-Za-z0-9\-\.]{{1,64}})?(\?.*)?'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--path',
        help='Directory containing subdirectories for each patient',
        default='.\\fhir\\Participant',

    )
    parser.add_argument(
        '-b',
        '--bin-size',
        help='Histogram bin size',
        default=5,
        type=int
    )
    parser.add_argument(
        '-d',
        '--debug',
        help='Show debug messages',
        action='store_const',
        dest='log_level',
        const=logging.DEBUG,
        default=logging.WARNING
    )
    parser.add_argument(
        '-f',
        '--file',
        help='config file running this module',
        default='',

    )

    return parser.parse_args()


def find_resource_files(directory):
    """Yield tuples of resource type (e.g. `SMOKING_STATUS`), full path of the
    resource results file obtained from the log files in each subdirectory and
    corresponding base FHIR URI. Resource type is determined by characters from
    the filename up to the first period.
    """
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue
        try:
            with open(os.path.join(path, 'log.json')) as f:
                log_data = json.load(f)
            logging.debug('Parsed log file in {}'.format(path))
            for query in log_data['query']:
                if query['status'] != 200:
                    continue
                filename = query['response']
                type_ = filename[:filename.index('.')]

                m = re.match(
                    BASE_URI_TEMPLATE.format(FILE_TYPE_MAPPING[type_]),
                    query['request']
                )
                base_uri = m.group('base_uri')

                logging.debug('Found {} of type {}'.format(
                    os.path.join(path, filename), type_
                ))
                yield type_, os.path.join(path, filename), base_uri
        except (IOError, ValueError, KeyError):
            # ignore if log file doesn't exist, doesn't parse as JSON or the
            # JSON doesn't have the keys we expect
            continue


def process_directory(directory):
    """Given a `SyncForScience` directory within a patient directory, collects
    total number of resources returned in each searchset by counting unique
    resource IDs for each JSON file present in the directory tree. Returns the
    base FHIR URI for the directory (assuming all entries in the log file
    originate from the same FHIR server) and a mapping of resource type to
    number of returned results.
    """
    logging.debug('Processing {}'.format(directory))

    uniques = defaultdict(set)
    base_uri = None
    for type_, path, resource_base_uri in find_resource_files(directory):
        if not base_uri:
            base_uri = resource_base_uri
        if type_ == 'PATIENT_DEMOGRAPHICS':
            logging.debug('Skipping {}'.format(path))
            continue  # do nothing with patient demographics for now
        try:
            with open(path) as f:
                data = json.load(f)
                logging.debug(
                    'Parsed {} as JSON'.format(path)
                )
        except ValueError:
            logging.debug(
                '{} could not be parsed as JSON'.format(path)
            )
            continue  # any non-JSON files will be ignored
        if 'entry' not in data:
            data['entry'] = list()  # no data

        # update set of unique resource IDs of the correct ResourceType
        uniques[type_].update(
            entry['resource']['id']
            for entry in data['entry']
            if entry['resource']['resourceType'] == FILE_TYPE_MAPPING[type_]
        )

    return base_uri, uniques

def data_in_directory(directory):
    """Given a `SyncForScience` directory within a patient directory, collects
    the data entries by resourceType
    """
    logging.debug('Processing {}'.format(directory))

    person = {}
    base_uri = None
    for type_, path, resource_base_uri in find_resource_files(directory):
        if not base_uri:
            base_uri = resource_base_uri
        try:
            with open(path) as f:
                data = json.load(f)
                logging.debug(
                    'Parsed {} as JSON'.format(path)
                )
        except ValueError:
            logging.debug(
                '{} could not be parsed as JSON'.format(path)
            )
            continue  # any non-JSON files will be ignored
        if 'entry' not in data:
            data['entry'] = list()  # no data

        if not type_ in person.keys():
            person[type_] = []
        for entry in data['entry']:
            person[type_].append(entry['resource'])

    return base_uri, person


def main():
    """Find patient S4S directories and compute statistics on the number of
    resources found for each resource type present stratified by base FHIR URI.
    Statistics include mean, median, max, min, and histogram data.
    """
    args = parse_arguments()
    logging.basicConfig(level=args.log_level)

    # mapping of base URI to another mapping of resource type to list of
    # resource counts for each patient
    total_counts = defaultdict(lambda: defaultdict(list))

    search_path = os.path.join(args.path, '*', 'SyncForScience')
    for directory in glob.glob(search_path):
        base_uri, dir_counts = process_directory(directory)
        for type_, count in dir_counts.items():
            total_counts[base_uri][type_].append(count)

    # TODO: pad each count list with 0s if necessary

    # output data structure
    results = dict()
    for base_uri, uri_counts in total_counts.items():
        uri_results = dict()
        for type_, counts in uri_counts.items():
            counts.sort()  # for median
            n = len(counts)
            summary = {
                'mean': sum(counts) / n,
                'median': counts[n // 2],  # not a true median, oops
                'min': counts[0],
                'max': counts[-1],
                'histogram': list()
            }

            # generate histogram data
            b = 0
            while True:
                # number of counts in the bin
                in_bin = sum(
                    b * args.bin_size <= x < (b + 1) * args.bin_size
                    for x in counts
                )
                if not in_bin:
                    b += 1
                    continue  # to save space, don't print empty bins
                summary['histogram'].append({
                    'bin_start': b * args.bin_size,
                    'bin_end': (b + 1) * args.bin_size - 1,  # inclusive
                    'count': in_bin
                })
                if (b + 1) * args.bin_size > counts[-1]:
                    break
                b += 1
            uri_results[type_] = summary
        results[base_uri] = uri_results
    return results


if __name__ == '__main__':
    print(json.dumps(main(), indent=2, sort_keys=True))

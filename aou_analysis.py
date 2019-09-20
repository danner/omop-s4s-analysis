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

import os
import pandas as pd
from fire import Fire
from collections import OrderedDict

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if basename == pattern:
                filename = os.path.join(root, basename)
                yield filename

# METRIC = []


def main(dir, metric):
    results = []

    for filename in find_files(dir, 'all_results.json'):

        
        res_df = pd.read_json(filename, typ='series')

        res_dict = OrderedDict()
        
        avg = 0.

        
        val = res_df[metric]
        res_dict[metric] = round(val * 100., 1)
        avg += val
        
        # res_dict['Avg'] = round((avg / len(XNLI_LANGUAGES))*100., 1)

        

        df = pd.DataFrame([res_dict])

        # df = pd.read_csv(filename)
        df.index = [os.path.basename(os.path.dirname(filename))] * len(df)
        results.append(df)

    results_df = pd.concat(results)
    sorted_results = results_df.sort_values(by=metric)
    sorted_results.to_csv(os.path.join(dir, 'all_result_sorted.csv'))

if __name__ == '__main__':
    Fire(main)
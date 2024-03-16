import os
import pandas as pd
from fire import Fire

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if basename == pattern:
                filename = os.path.join(root, basename)
                yield filename

def main(dir):
    results = []

    for filename in find_files(dir, 'all_result.csv'):
        df = pd.read_csv(filename)
        df.index = [os.path.basename(os.path.dirname(filename))] * len(df)
        results.append(df)

    results_df = pd.concat(results)
    sorted_results = results_df.sort_values(by='Avg')
    sorted_results.to_csv(os.path.join(dir, 'all_result_sorted.csv'))

if __name__ == '__main__':
    Fire(main)
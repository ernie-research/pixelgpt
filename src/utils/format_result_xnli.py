import os
import pandas as pd
import json
import fire

from collections import OrderedDict

XNLI_LANGUAGES = ["en", "ar", "bg", "de", "el", "fr", "hi", "ru", "es", "sw", "th", "tr", "ur", "vi", "zh"]

def main(output_dir: str):

    res_df = pd.read_json(os.path.join(output_dir, 'all_results.json'), typ='series')

    res_dict = OrderedDict()

    for lan in XNLI_LANGUAGES:
        key = 'test' + '_' + lan + '_' + 'accuracy'
        res_dict[key] = res_df[key]

    new_res_df = pd.DataFrame([res_dict])
    new_res_df.to_csv(os.path.join(output_dir, 'all_result.csv'), index=False)

if __name__ == '__main__':
    # output_dir = 'test'
    # main(output_dir)
    
    fire.Fire(main)
    
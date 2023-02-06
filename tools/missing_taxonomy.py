import argparse
import tdtax 
import os
import scope_phenom # https://github.com/bfhealy/scope-phenomenology
from scope.utils import read_parquet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

phenom = scope_phenom.taxonomy
sitewide = tdtax.taxonomy

# Recursive functions to perform a reverse lookup in taxonomy dictionaries

def trace_path(dct, key):
    if dct == key: 
        return [dct] 
    elif isinstance(dct, dict): 
        for k, v in dct.items(): 
            p = trace_path(v, key) 
            if p: 
                return [k] + p 
    elif isinstance(dct, list): 
        lst = dct 
        for i in range(len(lst)): 
            p = trace_path(lst[i], key) 
            if p: 
                return [str(i)] + p 

def get_class_path(trail, dct, key):
    stepdown = dct[trail[0]][int(trail[1])]
    cls = stepdown['class']
    if cls == key:
        return [cls]
    else:
        trail = trail[2:]
        p = get_class_path(trail, stepdown, key)
        return [cls] + p

def missing_taxonomy(parquet, mapper):
    # Read in golden dataset (downloaded from Fritz), mapper
    parquet_path = os.path.join(os.path.dirname(__file__), parquet)
    mapper_path = os.path.join(os.path.dirname(__file__), mapper)
    output_path = os.path.join(os.path.dirname(__file__), "golden_missing_labels.csv")
    gold_new = read_parquet(parquet_path)
    golden_dataset_mapper = pd.read_json(mapper_path)
    gold_map = golden_dataset_mapper.copy()
    none_cols = gold_map.loc['fritz_label'] == 'None'
    gold_map = gold_map.drop(columns=none_cols.index[none_cols.values])

    # Manipulate golden_dataset_mapper to flip keys and values
    gold_map = gold_map.transpose()
    gold_map.index.rename('trainingset_label', inplace=True)
    gold_map = gold_map.reset_index(drop=False).set_index('fritz_label')
    gold_dict = gold_map.transpose().to_dict()

    labels_gold = gold_new.set_index('obj_id')[gold_new.columns[1:54]]

    classes = labels_gold.columns.values.tolist()

    missing_df = pd.DataFrame({'obj_id':labels_gold.index, 'missing_descr':np.zeros(len(labels_gold),dtype=str)}).set_index('obj_id')

    values = []
    missing_items = []
    cnt = 0

    for index, row in labels_gold.iterrows():
        nonzero_vals = row[row>0].index.values
        for value in nonzero_vals:
            mapped_c = golden_dataset_mapper[value]['fritz_label']
            try:
                trail = trace_path(sitewide, mapped_c)
                class_path = get_class_path(trail, sitewide, mapped_c)
            except TypeError:
                trail = trace_path(phenom, mapped_c)
                class_path = get_class_path(trail, phenom, mapped_c)
            for item in class_path:
                if (item != mapped_c) & (item in classes):
                    mapped_item = gold_dict[item]['trainingset_label']
                    if labels_gold.loc[index, mapped_item] == 0:
                        cnt += 1
                        print(cnt, index, value, 'missing', mapped_item, ';')
                        values += [value]
                        missing_items += [mapped_item]
                        missing_df.loc[index, 'missing_descr'] = missing_df.loc[index, 'missing_descr'] + f"{value} missing {mapped_item};"
        missing_df.loc[index, 'missing_descr'] = missing_df.loc[index, 'missing_descr'][:-1]
    missing_df[missing_df['missing_descr']!=''].reset_index().to_csv(output_path,index=False)
    return None
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-parquet", type=str, help="path to parquet")
    parser.add_argument("-mapper", type=str, help="path to mapper")
    parser.add_argument(
        "-merge_features",
        type=bool,
        nargs='?',
        const=True,
        default=False,
        help="merge downloaded results with features from Kowalski",
    )
    parser.add_argument(
        "-features_catalog",
        type=str,
        default='ZTF_source_features_DR5',
        help="catalog of features on Kowalski",
    )
    
    args = parser.parse_args()

    missing_taxonomy(args.parquet, args.mapper)
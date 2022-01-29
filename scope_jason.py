import pandas as pd
import pdb
import json
from penquins import Kowalski
import numpy as np
from scope import Scope
import argparse
import csv

## load dataset
parser = argparse.ArgumentParser()
parser.add_argument("inputfile")
parser.add_argument("--id", type=int, default=1, help="group id on Fritz")
args = parser.parse_args()
with open(args.inputfile) as f:
    data = csv.reader(f)
    trainingset = pd.DataFrame(data)

# Kowalski
with open('password.txt', 'r') as f:
    password = f.read().splitlines()
G = Kowalski(username=password[0], password=password[1], host='gloria.caltech.edu', timeout=1000)

# get scores and data and combine
obj = Scope()
scores = obj.get_highscoring_objects(otype='vnv',condition='$or')

index = scores.index
condition = ((scores["vnv_dnn"]>0.95)&(scores['vnv_xgb']<=0.1)) | ((scores["vnv_dnn"]<=0.1)&(scores['vnv_xgb']>0.95))
disagreements = index[condition]
disagreeing_scores = scores.iloc[disagreements,:]

stats = obj.get_stats(disagreeing_scores['_id'])
data = pd.merge(disagreeing_scores,stats,left_on='_id',right_on='_id')
data['train'] = np.isin(data['_id'],trainingset['ztf_id'])
sample = data[~data['train']]

with open('token.txt', 'r') as f:
    token = f.read()

def upload():
    import requests

    def api(method, endpoint, data=None):
        headers = {'Authorization': f'token {token}'}
        response = requests.request(method, endpoint, json=data, headers=headers)
        return response


    headers={"Authorization": f"token {token}"}

    for index, row in sample.iterrows():
        #print(row)
        i = row['_id']
        period = row['period']
        
        # upload 
        json = { "catalog": "ZTF_sources_20210401",
                  "group_ids": [args.id]} 
        json['light_curve_ids'] = [int(i)]
        response = requests.post(
          url='https://fritz.science/api/archive',
          json=json,
          headers=headers,)

        # get objid
        try:
            obj_id = response.json()['data']['obj_id']
        except:
            continue

        # annotate
        url = 'https://fritz.science/api/sources/%s/annotations' %obj_id
        json = { "origin": "ML_DR5_d15_v1",
              "group_ids": [args.id]}
        
        Gaia = {'Plx': row['Gaia_EDR3__parallax'],'Mag_G': row['Gaia_EDR3__phot_g_mean_mag'],'Mag_Bp': row['Gaia_EDR3__phot_bp_mean_mag'],'Mag_Rp': row['Gaia_EDR3__phot_rp_mean_mag']}

        json['data'] = {'period': row['period'] , 
                        'vnv': {'vnv_dnn': np.round(row['vnv_dnn'],3),  'vnv_xgb': np.round(row['vnv_xgb'],3)},
                        'Gaia': Gaia}

        response = requests.post(
          url=url,
          json=json,
          headers=headers,)

upload()

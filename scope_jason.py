import pandas as pd
import pdb
import json
from penquins import Kowalski
import numpy as np

def get_highscoring_objects(otype='vnv',condition="$or",
    limit=0.9,limit_dnn=None,limit_xgb=None,):

    if limit_dnn == None:
        limit_dnn = limit
    if limit_xgb == None:
        limit_xgb = limit

    ### example
    q = {"query_type": "find",
             "query": {
                 "catalog": "ZTF_source_classifications_DR5",
                 "filter": { condition : [{'%s_xgb' %otype: { '$gt': limit_xgb }},
                                    {'%s_dnn' %otype: { '$gt': limit_dnn }}],
                           },
                 "projection": {}
             },
             "kwargs": {     }
             }

    r = G.query(q)

    return pd.DataFrame(r['data'])

def get_stats(ids):
    qs = [
           {"query_type": "find",
             "query": {
                 "catalog": "ZTF_source_features_DR5",
                 "filter": {'_id': i
                            },
                 "projection": {}
             },
             "kwargs": {     }
             }
        for i in ids
    ]
    rs = G.batch_query(qs, n_treads=32)

    return pd.DataFrame([s['data'][0] for s in rs])

# load dataset
trainingset = pd.read_csv('dataset.d15.csv')

# Kowalski
with open('password.txt', 'r') as f:
    password = f.read().splitlines()
G = Kowalski(username=password[0], password=password[1], host='gloria.caltech.edu', timeout=1000)

# get scores and data and combine
scores = get_highscoring_objects(otype='vnv',condition='$or')

index = scores.index
condition = ((scores["vnv_dnn"]>0.95)&(scores['vnv_xgb']<=0.1)) | ((scores["vnv_dnn"]<=0.1)&(scores['vnv_xgb']>0.95))
disagreements = index[condition]
disagreeing_scores = scores.iloc[disagreements,:]

stats = get_stats(disagreeing_scores['_id'])
data = pd.merge(disagreeing_scores,stats,left_on='_id',right_on='_id')
data['train'] = np.isin(data['_id'],trainingset['ztf_id'])
sample = data[~data['train']]

def upload():
    import requests
    token = "9f78fcea-61f0-4b71-8b9f-d1b1b573fd4d" # Jason's Fritz token

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
                  "group_ids": [371]} #group id for the upload location
        json['light_curve_ids'] = [int(i)]
        response = requests.post(
          url='https://fritz.science/api/archive',
          json=json,
          headers=headers,)

        # get objid
        try:
            obj_id = response.json()['data']['obj_id']
        except:
            print('failed to upload target')
            continue

        print(obj_id)

        # annotate
        url = 'https://fritz.science/api/sources/%s/annotations' %obj_id
        json = { "origin": "ML_DR5_d15_v1",
              "group_ids": [348]}
        
        Gaia = {'Plx': row['Gaia_EDR3__parallax'],'Mag_G': row['Gaia_EDR3__phot_g_mean_mag'],'Mag_Bp': row['Gaia_EDR3__phot_bp_mean_mag'],'Mag_Rp': row['Gaia_EDR3__phot_rp_mean_mag']}

        json['data'] = {'period': row['period'] , 
                        'vnv': {'vnv_dnn': np.round(row['vnv_dnn'],3),  'vnv_xgb': np.round(row['vnv_xgb'],3)},
                        'Gaia': Gaia}

        response = requests.post(
          url=url,
          json=json,
          headers=headers,)

upload()

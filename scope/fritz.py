import pandas as pd


def get_highscoring_objects(
    G,
    otype='vnv',
    condition="$or",
    limit=0.9,
    limit_dnn=None,
    limit_xgb=None,
):

    if limit_dnn is None:
        limit_dnn = limit
    if limit_xgb is None:
        limit_xgb = limit

    # example
    q = {
        "query_type": "find",
        "query": {
            "catalog": "ZTF_source_classifications_DR5",
            "filter": {
                condition: [
                    {'%s_xgb' % otype: {'$gt': limit_xgb}},
                    {'%s_dnn' % otype: {'$gt': limit_dnn}},
                ],
            },
            "projection": {},
        },
        "kwargs": {},
    }

    r = G.query(q)

    return pd.DataFrame(r['data'])


def get_stats(G, ids):
    qs = [
        {
            "query_type": "find",
            "query": {
                "catalog": "ZTF_source_features_DR5",
                "filter": {'_id': i},
                "projection": {},
            },
            "kwargs": {},
        }
        for i in ids
    ]
    rs = G.batch_query(qs, n_treads=32)

    return pd.DataFrame([s['data'][0] for s in rs])

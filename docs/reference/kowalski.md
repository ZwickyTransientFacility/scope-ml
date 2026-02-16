# Kowalski Query Reference

## Overview

[Kowalski](https://github.com/skyportal/kowalski) is the database system used by SCoPe to store and query ZTF light curves, features, and classifications. SCoPe interacts with three Kowalski instances:

- **kowalski.caltech.edu** — primary instance with ZTF alerts, source catalogs, and cross-match data
- **gloria.caltech.edu** — hosts ZTF source features and classifications (DR3–DR5)
- **melman.caltech.edu** — hosts the latest ZTF source features and classifications (DR16+)

All interactions use the [penquins](https://github.com/skyportal/penquins) Python client.

## Authentication

### First-Time Setup

Authenticate with username and password to generate tokens:

```python
from penquins import Kowalski

kowalski = Kowalski(
    username="your_username",
    password="your_password",
    host="kowalski.caltech.edu",
)

gloria = Kowalski(
    username="your_username",
    password="your_password",
    host="gloria.caltech.edu",
)

melman = Kowalski(
    username="your_username",
    password="your_password",
    host="melman.caltech.edu",
)

# Generate tokens (only needed once)
k_token = kowalski.authenticate()
g_token = gloria.authenticate()
m_token = melman.authenticate()
```

Save the printed tokens to your `config.yaml` file for future use.

### Token-Based Authentication

After obtaining tokens, create instances without re-authenticating:

```python
from penquins import Kowalski

tokens = {
    "kowalski": "your_kowalski_token",
    "gloria": "your_gloria_token",
    "melman": "your_melman_token",
}

hosts = ["kowalski", "gloria", "melman"]

instances = {
    host: {
        "protocol": "https",
        "port": 443,
        "host": f"{host}.caltech.edu",
        "token": tokens[host],
    }
    for host in hosts
}

kowalski_instances = Kowalski(instances=instances, timeout=610)
```

### Verify Connections

```python
for host in hosts:
    print(host, kowalski_instances.ping(name=host))
```

## Common Queries

### List Available Catalogs

```python
query = {
    "query_type": "info",
    "query": {
        "command": "catalog_names",
    },
}

# Specify instance name for queries without a catalog
response = kowalski_instances.query(query=query, name="melman")
catalogs = response.get("melman").get("data")
```

### Get Document Count

```python
qry = kowalski_instances.query({
    "query_type": "estimated_document_count",
    "query": {
        "catalog": "ZTF_source_classifications_DR16",
    },
})

count = qry.get("melman").get("data")
```

### Count Documents Matching a Filter

```python
qry = kowalski_instances.query({
    "query_type": "count_documents",
    "query": {
        "catalog": "ZTF_source_classifications_DR16",
        "filter": {
            "$or": [
                {"vnv_xgb": {"$gt": 0.99}},
                {"vnv_dnn": {"$gt": 0.99}},
            ]
        },
        "kwargs": {
            "max_time_ms": 600000,
        },
    },
})

n = qry.get("melman").get("data")
```

### Loop Over All Instances

When a catalog exists on multiple instances, queries are automatically distributed. Loop over instance names to aggregate results:

```python
total = 0
for instance_name in qry:
    result = qry.get(instance_name).get("data")
    total += result
    print(instance_name, result)
```

## Batch Queries

Large result sets must be fetched in batches to avoid timeouts. Use `skip` and `limit` to paginate:

```python
import numpy as np
import pandas as pd

batch_size = 10000
n_batches = int(np.ceil(nlightcurves / batch_size))
df_all = []

for n in range(n_batches):
    qry = kowalski_instances.query({
        "query_type": "find",
        "query": {
            "catalog": "ZTF_source_classifications_DR16",
            "filter": {
                "$or": [
                    {"vnv_xgb": {"$gt": 0.99}},
                    {"vnv_dnn": {"$gt": 0.99}},
                ]
            },
        },
        "kwargs": {
            "skip": int(n * batch_size),
            "limit": int(batch_size),
            "max_time_ms": 600000,
        },
    })
    data = qry.get("melman").get("data")
    df_all.append(pd.DataFrame.from_records(data))

df_scores = pd.concat(df_all).reset_index(drop=True)
```

## Fetching Features

Once you have a list of source IDs, fetch their features from the features catalog:

```python
source_ids = df_scores["_id"].values.tolist()
features_catalog = "ZTF_source_features_DR16"
limit = 1000

idx = 0
df_collection = []

while True:
    query = {
        "query_type": "find",
        "query": {
            "catalog": features_catalog,
            "filter": {
                "_id": {"$in": source_ids[idx * limit : (idx + 1) * limit]}
            },
        },
    }
    response = kowalski_instances.query(query=query)
    source_data = response.get("melman").get("data")
    df_collection.append(pd.DataFrame.from_records(source_data))

    if ((idx + 1) * limit) >= len(source_ids):
        break
    idx += 1

df_features = pd.concat(df_collection, axis=0)
```

## Merging Scores and Features

```python
df_merged = pd.merge(df_scores, df_features, on="_id")

# Save to parquet
from scope.utils import write_parquet
write_parquet(df_merged, "merged_scores_features.parquet")
```

## Key Catalogs

| Catalog | Instance | Description |
|---------|----------|-------------|
| `ZTF_source_classifications_DR16` | melman | Latest source classifications |
| `ZTF_source_features_DR16` | melman | Latest source features (193 columns) |
| `ZTF_source_classifications_DR5` | gloria, melman | DR5 classifications |
| `ZTF_source_features_DR5` | gloria | DR5 features |
| `ZTF_sources_20230309` | melman | ZTF light curve sources |
| `Gaia_EDR3` | all | Gaia Early Data Release 3 |
| `PS1_DR1` | all | Pan-STARRS Data Release 1 |
| `AllWISE` | all | AllWISE catalog |

import numpy as np
from penquins import Kowalski
from scope.utils import split_dict


def get_ztf_alert_stats(
    id_dct: dict,
    kowalski_instances: Kowalski,
    catalog: str = 'ZTF_alerts',
    radius_arcsec: float = 2.0,
    limit: int = 10000,
    Ncore: int = 1,
):
    """
    Get n_ztf_alerts and mean_ztf_alert_braai features

    :param id_dct: one quadrant's worth of id-coordinate pairs (dict)
    :param kowalski_instances: authenticated instances of Kowalski databases
    :param catalog: name of alert catalog to use (str)
    :param radius_arcsec: size of cone within which to query alerts (float)
    :param limit: batch size of kowalski_instances queries (int)
    :param Ncore: number of cores for parallel queries

    :return alert_stats_dct: Dictionary containing n_ztf_alerts and mean_ztf_alert_braai for each source ID
    """
    limit *= Ncore

    ids = [x for x in id_dct]

    n_sources = len(id_dct)
    if n_sources % limit != 0:
        n_iterations = n_sources // limit + 1
    else:
        n_iterations = n_sources // limit
    alert_results_dct = {}

    print(f'Querying {catalog} catalog in batches...')
    for i in range(0, n_iterations):
        print(f"Iteration {i+1} of {n_iterations}...")
        id_slice = [x for x in id_dct.keys()][
            i * limit : min(n_sources, (i + 1) * limit)
        ]

        radec_geojson = np.array(
            [id_dct[x]['radec_geojson']['coordinates'] for x in id_slice]
        ).transpose()

        # Need to add 180 -> no negative RAs
        radec_geojson[0, :] += 180.0
        radec_dict = dict(zip(id_slice, radec_geojson.transpose().tolist()))

        # Split dictionary for parallel querying
        radec_split_list = [lst for lst in split_dict(radec_dict, Ncore)]

        # Only keep non-empty dictionaries
        # (Needed for small lists of sources compared to Ncore)
        radec_split_list = [x for x in radec_split_list if len(x) > 0]

        queries = [
            {
                "query_type": "cone_search",
                "query": {
                    "object_coordinates": {
                        "radec": dct,
                        "cone_search_radius": radius_arcsec,
                        "cone_search_unit": 'arcsec',
                    },
                    "catalogs": {
                        catalog: {
                            "filter": {},
                            "projection": {"classifications.braai": 1},
                        }
                    },
                    "filter": {},
                },
            }
            for dct in radec_split_list
        ]

        responses = kowalski_instances.query(
            queries=queries, use_batch_query=True, max_n_threads=Ncore
        )

        for name in responses.keys():
            if len(responses[name]) > 0:
                response_list = responses[name]
                for response in response_list:
                    if response.get("status", "error") == "success":
                        alert_results = response.get("data").get(catalog)
                        if alert_results is not None:
                            alert_results_dct.update(alert_results)

    alert_stats_dct = {}

    for id in ids:
        n_ztf_alerts = len(alert_results_dct[str(id)])
        if n_ztf_alerts > 0:
            mean_ztf_alert_braai = np.nanmean(
                [
                    y
                    for x in alert_results_dct[str(id)]
                    for y in x['classifications'].values()
                ]
            )
        else:
            mean_ztf_alert_braai = 0.0
        alert_stats_dct[id] = {
            'n_ztf_alerts': n_ztf_alerts,
            'mean_ztf_alert_braai': mean_ztf_alert_braai,
        }

    return alert_stats_dct

import numpy as np
from penquins import Kowalski


def xmatch(
    id_dct: dict,
    kowalski_instance: Kowalski,
    catalog_info: dict,
    radius_arcsec: float = 2.0,
    limit: int = 10000,
):
    """
    Batch cross-match external catalogs by position

    :param id_dct: one quadrant's worth of id-coordinate pairs (dict)
    :param kowalski_instance: authenticated instance of a Kowalski database
    :param catalog_info: nested dict containing catalog names, filters and projections (see config.yaml)
    :param xmatch_radius_arcsec: size of cone within which to match a queried source with an input source.
    :param limit: batch size of kowalski_instance queries (int)

    :return id_dct_external: id_dct updated with external xmatch values
    """

    n_sources = len(id_dct)
    if n_sources % limit != 0:
        n_iterations = n_sources // limit + 1
    else:
        n_iterations = n_sources // limit
    ext_results_dct = {}

    print('Querying crossmatch catalogs in batches...')
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

        # Get external catalog data
        query = {
            "query_type": "cone_search",
            "query": {
                "object_coordinates": {
                    "radec": dict(zip(id_slice, radec_geojson.transpose().tolist())),
                    "cone_search_radius": radius_arcsec,
                    "cone_search_unit": 'arcsec',
                },
                "catalogs": catalog_info,
                "filter": {},
            },
        }
        q = kowalski_instance.query(query)

        ext_results = q['data']
        ext_results_dct.update(ext_results)

    id_dct_external = id_dct.copy()
    print('Iterating through results and assigning xmatch features to sources...')
    for catalog in catalog_info.keys():
        cat_results = ext_results_dct[catalog]
        cat_projection_names = [x for x in catalog_info[catalog]['projection'].keys()]
        for id in id_dct_external.keys():
            ext_values = cat_results[str(id)]
            if len(ext_values) > 0:
                ext_values = ext_values[0]
                for val_name in cat_projection_names:
                    try:
                        id_dct_external[id][f'{catalog}__{val_name}'] = ext_values[
                            val_name
                        ]
                    except KeyError:
                        id_dct_external[id][f'{catalog}__{val_name}'] = np.nan
            else:
                for val_name in cat_projection_names:
                    id_dct_external[id][f'{catalog}__{val_name}'] = np.nan

    return id_dct_external

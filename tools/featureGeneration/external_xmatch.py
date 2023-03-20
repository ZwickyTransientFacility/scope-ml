import numpy as np
from penquins import Kowalski
from astropy.coordinates import SkyCoord
from scope.utils import split_dict


def xmatch(
    id_dct: dict,
    kowalski_instance: Kowalski,
    catalog_info: dict,
    radius_arcsec: float = 2.0,
    limit: int = 10000,
    Ncore: int = 1,
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
    limit *= Ncore

    n_sources = len(id_dct)
    if n_sources % limit != 0:
        n_iterations = n_sources // limit + 1
    else:
        n_iterations = n_sources // limit
    ext_results_dct = {}
    for c in catalog_info.keys():
        ext_results_dct[c] = {}

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
        radec_dict = dict(zip(id_slice, radec_geojson.transpose().tolist()))

        # Store desired external features for later
        cat_projection_names = {}
        for catalog in catalog_info.keys():
            cat_projection_names[catalog] = [
                x for x in catalog_info[catalog]['projection'].keys()
            ]

        # Add coordinates.radec_geojson to each catalog projection
        [
            catalog_info[x]['projection'].update({'coordinates.radec_geojson': 1})
            for x in catalog_info.keys()
        ]

        if Ncore > 1:
            # Split dictionary for parallel querying
            radec_split_list = [lst for lst in split_dict(radec_dict, Ncore)]

            queries = [
                {
                    "query_type": "cone_search",
                    "query": {
                        "object_coordinates": {
                            "radec": dct,
                            "cone_search_radius": radius_arcsec,
                            "cone_search_unit": 'arcsec',
                        },
                        "catalogs": catalog_info,
                        "filter": {},
                    },
                }
                for dct in radec_split_list
            ]
            q = kowalski_instance.batch_query(queries, n_treads=Ncore)
            for batch_result in q:
                ext_results = batch_result['data']
                for c in catalog_info.keys():
                    ext_results_dct[c].update(ext_results[c])

        else:
            # Get external catalog data
            query = {
                "query_type": "cone_search",
                "query": {
                    "object_coordinates": {
                        "radec": radec_dict,
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
    print(len(ext_results_dct))
    for catalog in catalog_info.keys():
        cat_results = ext_results_dct[catalog]
        print(len(cat_results))
        print(cat_results.keys())
        for id in id_dct_external.keys():
            ext_values = cat_results[str(id)]

            if len(ext_values) > 0:
                if len(ext_values) == 1:
                    ext_values = ext_values[0]
                else:
                    # If more than one source is matched, choose the closest
                    ra_input, dec_input = id_dct_external[id]['radec_geojson'][
                        'coordinates'
                    ]
                    input_SC = SkyCoord(
                        ra_input + 180.0, dec_input, unit=['deg', 'deg']
                    )
                    ras = []
                    decs = []

                    for entry in ext_values:
                        ra_match, dec_match = entry['coordinates']['radec_geojson'][
                            'coordinates'
                        ]
                        ras += [ra_match]
                        decs += [dec_match]

                    match_SC = SkyCoord(
                        ra_match + 180.0, dec_match, unit=['deg', 'deg']
                    )
                    seps_argmin = np.argmin(input_SC.separation(match_SC))
                    ext_values = ext_values[seps_argmin]

                for val_name in cat_projection_names[catalog]:
                    try:
                        id_dct_external[id][f'{catalog}__{val_name}'] = ext_values[
                            val_name
                        ]
                    except KeyError:
                        id_dct_external[id][f'{catalog}__{val_name}'] = np.nan
            else:
                for val_name in cat_projection_names[catalog]:
                    id_dct_external[id][f'{catalog}__{val_name}'] = np.nan

    return id_dct_external

#!/usr/bin/env python
import argparse
import pandas as pd
from scope.fritz import api
import warnings
import numpy as np
from tools.get_features import get_features
import os

NUM_PER_PAGE = 500
CHECKPOINT_NUM = 500


def organize_source_data(src: pd.DataFrame):
    id = src['id']
    ra = src['ra']
    dec = src['dec']

    data_classes = src['classifications']
    cls_list = ''
    prb_list = ''
    for entry in data_classes:
        cls = entry['classification']
        prb = entry['probability']
        cls_list += cls + ';'  # same format as download from Fritz frontend
        prb_list += str(prb) + ';'

    cls_list = cls_list[:-1]  # remove trailing semicolon
    prb_list = prb_list[:-1]

    # loop through annotations, checking for periods
    data_annot = src['annotations']
    origin_list = ''
    period_list = ''
    id_origin_list = ''
    id_list = ''
    for entry in data_annot:
        annot_origin = entry['origin']
        annot_data = entry['data']
        annot_name = [x for x in annot_data.keys()]

        for n in annot_name:
            # if period is found, add to list
            if n == 'period':
                origin_list += annot_origin + ';'
                period_list += str(annot_data[n]) + ';'

            elif n == 'ztf_id':
                id_origin_list += annot_origin + ';'
                id_list += str(annot_data[n]) + ';'

    origin_list = origin_list[:-1]
    period_list = period_list[:-1]
    id_origin_list = id_origin_list[:-1]
    id_list = id_list[:-1]

    dct = {}
    (
        dct['obj_id'],
        dct['ra'],
        dct['dec'],
        dct['classification'],
        dct['probability'],
        dct['period_origin'],
        dct['period'],
        dct['ztf_id_origin'],
        dct['ztf_id'],
    ) = (
        id,
        ra,
        dec,
        cls_list,
        prb_list,
        origin_list,
        period_list,
        id_origin_list,
        id_list,
    )

    return dct


def merge_sources_features(
    sources,
    features_catalog,
    features_limit=1000,
    mapper_name='golden_dataset_mapper.json',
    output_dir='fritzDownload',
    output_filename='merged_classifications_features.csv',
):

    outpath = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(outpath, exist_ok=True)

    # Drop rows with no ZTF id
    no_ztf_id = np.sum((sources.isna()['ztf_id']) | (sources['ztf_id'] == ''))
    if no_ztf_id > 0:
        print(f'Dropping {no_ztf_id} sources without a ZTF id.')
        sources = sources[
            ~((sources.isna()['ztf_id']) | (sources['ztf_id'] == ''))
        ].reset_index(drop=True)

    # Drop rows with duplicate ZTF ids (keep first instance)
    dup_ztf_id = np.sum(sources.duplicated('ztf_id'))
    if dup_ztf_id > 0:
        print(f'Dropping {dup_ztf_id} sources with duplicate ZTF ids.')
        sources = sources.drop_duplicates('ztf_id').reset_index(drop=True)

    features_ztf_dr = features_catalog.split('_')[-1]
    source_ids = sources['ztf_id'].values.astype(int).tolist()

    # Open golden dataset mapper
    mapper_dir = os.path.dirname(__file__)
    mapper_path = os.path.join(mapper_dir, mapper_name)
    gold_map = pd.read_json(mapper_path)

    # Drop columns with no equivalent in Fritz
    none_cols = gold_map.loc['fritz_label'] == 'None'
    gold_map = gold_map.drop(columns=none_cols.index[none_cols.values])
    classes = gold_map.columns

    # Manipulate golden_dataset_mapper to flip keys and values
    gold_map = gold_map.transpose()
    gold_map.index.rename('trainingset_label', inplace=True)
    gold_map = gold_map.reset_index(drop=False).set_index('fritz_label')
    gold_dict = gold_map.transpose().to_dict()

    # Create empty dataframe for source classifications
    expanded_sources = pd.DataFrame(
        columns=np.concatenate([['obj_id'], classes, ['ztf_id']])
    )
    source_dict_list = []

    for index, row in sources.iterrows():
        gold_dict_specific = gold_dict.copy()

        # Initiate dictonary to concatenate with existing dataframe
        source_dict = {}
        obj_id = row['obj_id']

        # Assign Fritz object id
        source_dict['obj_id'] = obj_id

        classifications = []
        completed_classifications = []

        if not row.isna()['classification']:
            classifications = row['classification'].split(';')
            probabilities = row['probability'].split(';')

        # Assign given probability values for each Fritz classification
        for i in range(len(classifications)):
            cls = classifications[i]
            if cls not in completed_classifications:
                trainingset_label = gold_dict_specific[cls]['trainingset_label']
                gold_dict_specific.pop(cls)
                completed_classifications += [cls]
                source_dict[trainingset_label] = probabilities[i]

        # Assign zero probability for remaining labels
        for remaining_entry in gold_dict_specific:
            if remaining_entry not in completed_classifications:
                trainingset_label = gold_dict_specific[remaining_entry][
                    'trainingset_label'
                ]
                source_dict[trainingset_label] = 0.0

        # Assign ztf id
        if not row.isna()['ztf_id']:
            ztf_id_origin_dr = row['ztf_id_origin'].split('_')[-1]

        # Check if Fritz ZTF DR number is the same as features catalog
        ztf_id = row['ztf_id']
        if ztf_id_origin_dr == features_ztf_dr:
            source_dict['ztf_id'] = int(ztf_id)
        else:
            raise ValueError(
                'ZTF data release numbers do not match between Fritz and features catalog.'
            )

        source_dict_list += [source_dict]

    # Create dataframe
    expanded_sources = pd.concat([expanded_sources, pd.DataFrame(source_dict_list)])

    # Query Kowalski
    print('Getting features...')
    df, dmdt = get_features(
        source_ids=source_ids,
        features_catalog=features_catalog,
        limit=features_limit,
        write_results=False,
    )
    df['ztf_id'] = df['_id']
    df = df.drop('_id', axis=1)

    # Merge on ZTF id
    merged_set = pd.merge(expanded_sources, df, on='ztf_id')

    # Make ztf_id last column in dataframe
    ztf_id_col = merged_set.pop('ztf_id')
    merged_set['ztf_id'] = ztf_id_col

    filepath = os.path.join(outpath, output_filename)
    merged_set.to_csv(filepath, index=False)
    return merged_set


def download_classification(
    file: str,
    group_ids: list,
    start: int,
    merge_features: bool,
    features_catalog: str,
    features_limit: int,
    mapper_name: str = 'golden_dataset_mapper.json',
    output_dir: str = 'fritzDownload',
    output_filename: str = 'merged_classifications_features.csv',
):
    """
    Download labels from Fritz
    :param file: CSV file containing obj_id column or "parse" to query by group ids (str)
    :param group_ids: target group ids on Fritz for download (list)
    :param start: page number to start downloading data from (int)
    :param merge_features: if True,  query Kowalski for features to merge (bool)
    :param features_catalog: catalog name to query for features (str)
    :param features_limit: maximum number of sources to query for features per loop (int)
    """

    dict_list = []

    outpath = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(outpath, exist_ok=True)

    filename = (
        os.path.basename(file).removesuffix('.csv') + '_fritzDownload' + '.csv'
    )  # rename file
    filepath = os.path.join(outpath, filename)

    if file in ["parse", 'Parse', 'PARSE']:
        if group_ids is None:
            raise ValueError('Specify group_ids to query Fritz.')
        response = api(
            "GET",
            "/api/sources",
            {"group_ids": group_ids, "numPerPage": NUM_PER_PAGE},
        )
        source_data = response.json().get("data")

        # determine number of pages
        allMatches = source_data['totalMatches']
        nPerPage = source_data['numPerPage']
        pages = int(np.ceil(allMatches / nPerPage))

        if start != 0:
            filename = (
                filename.removesuffix('.csv') + f'_continued_page_{start}' + '.csv'
            )
            filepath = os.path.join(outpath, filename)
            print('Downloading sources...')
        else:
            start = 1
            print(f'Downloading {allMatches} sources...')

        # iterate over all pages in results
        for pageNum in range(start, pages + 1):
            print(f'Page {pageNum} of {pages}...')
            page_response = api(
                "GET",
                '/api/sources',
                {
                    "group_ids": group_ids,
                    'numPerPage': NUM_PER_PAGE,
                    'pageNumber': pageNum,
                },  # page numbers start at 1
            )
            page_data = page_response.json().get('data')

            for src in page_data['sources']:
                dct = organize_source_data(src)
                dict_list += [dct]

            # create dataframe from query results
            sources = pd.json_normalize(dict_list)
            sources.to_csv(filepath, index=False)
            print(f'Saved page {pageNum}.')

        if not merge_features:
            return sources
        else:
            merged_sources = merge_sources_features(
                sources,
                features_catalog,
                features_limit,
                mapper_name,
                output_dir,
                output_filename,
            )
            return merged_sources

    else:
        # read in CSV file
        sources = pd.read_csv(file)
        if start != 0:
            # continue from checkpoint
            sources = sources[start:]
            filename = (
                filename.removesuffix('.csv') + f'_continued_index_{start}' + '.csv'
            )
            filepath = os.path.join(outpath, filename)

        columns = sources.columns

        missing_col_count = 0
        for colname in [
            'classification',
            'probability',
            'period_origin',
            'period',
            'ztf_id_origin',
            'ztf_id',
        ]:
            if colname not in columns:
                missing_col_count += 1
        if missing_col_count > 0:

            # add obj_id column if not passed in
            if 'obj_id' in columns:
                search_by_obj_id = True
                no_obj_id = 0
            else:
                sources["obj_id"] = None
                search_by_obj_id = False
                no_obj_id = 1

            if ('ra' in columns) & ('dec' in columns):
                racol = sources['ra']
                decol = sources['dec']
                sources.drop(['ra', 'dec'], axis=1, inplace=True)
                no_ra_dec = 0

            if no_obj_id:
                if no_ra_dec:
                    raise KeyError(
                        'Please provide either obj_id or ra and dec columns.'
                    )

            dct_list = []
            for index, row in sources.iterrows():
                # query objects, starting with obj_id
                data = []
                if search_by_obj_id:
                    obj_id = row.obj_id
                    response = api("GET", '/api/sources/%s' % obj_id)
                    data = response.json().get("data")
                    if len(data) == 0:
                        warnings.warn(
                            'No results from obj_id search - querying by ra/dec.'
                        )
                    else:
                        src = data

                # continue with coordinate search if obj_id unsuccsessful
                if len(data) == 0:
                    # query by ra/dec to get object id
                    ra, dec = racol.loc[index], decol.loc[index]
                    response = api(
                        "GET",
                        f"/api/sources?&ra={ra}&dec={dec}&radius={2/3600}",
                    )
                    data = response.json().get("data")
                    obj_id = None
                    if data["totalMatches"] > 0:
                        src = data["sources"][0]
                        obj_id = src["id"]
                        sources.at[index, 'obj_id'] = obj_id

                print(f"object {index} id:", obj_id)

                # if successful search, get and save labels/probabilities/period annotations to sources dataframe
                if obj_id is not None:
                    dct = organize_source_data(src)
                    dct_list += [dct]

                    # occasional checkpoint at specified number of sources
                    if (index + 1) % CHECKPOINT_NUM == 0:
                        sources_chkpt = pd.merge(
                            sources, pd.json_normalize(dct_list), on='obj_id'
                        )
                        sources_chkpt.to_csv(filepath, index=False)
                        print(f'Saved checkpoint at index {index}.')

                else:
                    warnings.warn(f'Unable to find source {index} on Fritz.')

            # final save
            sources = pd.merge(sources, pd.json_normalize(dct_list), on='obj_id')
            print('Saving all sources.')
            sources.to_csv(filepath, index=False)

        if not merge_features:
            return sources
        else:
            merged_sources = merge_sources_features(
                sources,
                features_catalog,
                features_limit,
                mapper_name,
                output_dir,
                output_filename,
            )
            return merged_sources


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="dataset")
    parser.add_argument("-group_ids", type=int, nargs='+', help="list of group ids")
    parser.add_argument(
        "-start", type=int, default=0, help="start page/index for continued download"
    )
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
        help="catalog of features on Kowalksi",
    )

    parser.add_argument(
        "-features_limit",
        type=int,
        default=1000,
        help="Maximum number of sources queried for features per loop",
    )

    parser.add_argument(
        "-mapper_name",
        type=str,
        default='golden_dataset_mapper.json',
        help="Filename of classification mapper",
    )

    parser.add_argument(
        "-output_dir",
        type=str,
        default='fritzDownload',
        help="Name of directory to save downloaded files",
    )

    parser.add_argument(
        "-output_filename",
        type=str,
        default='merged_classifications_features.csv',
        help="Name of file containing merged classifications and features",
    )

    args = parser.parse_args()

    # download object classifications in the file
    download_classification(
        args.file,
        args.group_ids,
        args.start,
        args.merge_features,
        args.features_catalog,
        args.features_limit,
        args.mapper_name,
        args.output_dir,
        args.output_filename,
    )

#!/usr/bin/env python
import argparse
import pandas as pd
import scope
from scope.fritz import api
from scope.utils import read_hdf, write_hdf, read_parquet, write_parquet
import warnings
import numpy as np
from tools.get_features import get_features
import os
from datetime import datetime

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
    output_filename='merged_classifications_features',
    output_format='.parquet',
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

    # Manipulate golden_dataset_mapper to flip keys and values
    gold_map = gold_map.transpose()
    gold_map.index.rename('trainingset_label', inplace=True)
    gold_map = gold_map.reset_index(drop=False).set_index('fritz_label')
    gold_dict = gold_map.transpose().to_dict()

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
                try:
                    trainingset_label = gold_dict_specific[cls]['trainingset_label']
                    gold_dict_specific.pop(cls)
                    source_dict[trainingset_label] = float(probabilities[i])
                except KeyError:
                    print(f'Key {cls} not in dataset mapper.')
                    continue
                completed_classifications += [cls]

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
    expanded_sources = pd.DataFrame(source_dict_list)

    # Query Kowalski
    print('Getting features...')
    df, dmdt = get_features(
        source_ids=source_ids,
        features_catalog=features_catalog,
        limit=features_limit,
        write_results=False,
    )
    df['ztf_id'] = df['_id']
    df = df.drop(['_id'], axis=1)

    # Merge on ZTF id
    merged_set = pd.merge(expanded_sources, df, on='ztf_id')
    merged_set.attrs = sources.attrs

    # Make ztf_id last column in dataframe
    ztf_id_col = merged_set.pop('ztf_id')
    merged_set['ztf_id'] = ztf_id_col

    # Add more metadata
    utcnow = datetime.utcnow()
    start_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")
    merged_set.attrs['features_download_dateTime_utc'] = start_dt
    merged_set.attrs['features_ztf_dataRelease'] = features_ztf_dr

    filepath = os.path.join(outpath, output_filename + output_format)
    if output_format == '.csv':
        merged_set.to_csv(filepath, index=False)
    elif output_format == '.h5':
        warnings.warn(
            'Coordinates and dmdt features get pickled and put in separate datasets, not supported by hdf5 format.'
        )
        coordinates = merged_set['coordinates']
        dmdt = merged_set['dmdt']
        merged_set = merged_set.drop(['coordinates', 'dmdt'], axis=1)
        # Write hdf5 file
        # Caution: coordinates and dmdt features get pickled and put in separate datasets, not supported by hdf5 format.
        # File will be comparable in size to csv
        with pd.HDFStore(filepath, mode='w') as store:
            store.put('df', merged_set, format='table')
            store.put('coordinates', coordinates)
            store.put('dmdt', dmdt)
            store.get_storer('df').attrs.metadata = merged_set.attrs
    else:
        write_parquet(merged_set, filepath)

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
    output_filename: str = 'merged_classifications_features',
    output_format: str = '.parquet',
):
    """
    Download labels from Fritz
    :param file: CSV, hdf5 or parquet file containing obj_id column or "parse" to query by group ids (str)
    :param group_ids: target group ids on Fritz for download (list)
    :param start: page number to start downloading data from (int)
    :param merge_features: if True,  query Kowalski for features to merge (bool)
    :param features_catalog: catalog name to query for features (str)
    :param features_limit: maximum number of sources to query for features per loop (int)
    :param output_dir: directory to write output merged features file (str)
    :param output_filename: name of output merged features file (str)
    :param output_format: format of output merged features file (str)
    """

    dict_list = []

    # Check for appropriate output format
    output_file_extension = os.path.splitext(output_filename)[-1]

    # If parquet, h5 or csv extension specified in output_filename, use that format for saving
    output_format = (
        output_format
        if output_file_extension not in ['.parquet', '.h5', '.csv']
        else output_file_extension
    )

    if output_format in [
        '.parquet',
        'parquet',
        '.parq',
        'parq',
        '.PARQUET',
        'PARQUET',
        '.PARQ',
        'PARQ',
    ]:
        output_format = '.parquet'
        print('Using .parquet extension for saved files.')
    elif output_format in ['.h5', 'h5', '.H5', 'H5', '.hdf5', 'hdf5', '.HDF5', 'HDF5']:
        output_format = '.h5'
        print('Using .h5 extension for saved files.')
    elif output_format in ['.csv', 'csv', '.CSV', 'CSV']:
        output_format = '.csv'
        print('Using .csv extension for saved files.')
    else:
        raise ValueError('Output format must be parquet, hdf5 or csv.')

    # If user puts extension in filename, remove it for consistency
    if (
        (output_filename.endswith('.csv'))
        | (output_filename.endswith('.h5'))
        | (output_filename.endswith('.parquet'))
    ):
        output_filename = os.path.splitext(output_filename)[0]

    outpath = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(outpath, exist_ok=True)

    filename = (
        os.path.splitext(os.path.basename(file))[0] + '_fritzDownload' + output_format
    )  # rename file
    filepath = os.path.join(outpath, filename)

    # Get code version and current date/time for metadata
    code_version = scope.__version__
    utcnow = datetime.utcnow()
    start_dt = utcnow.strftime("%Y-%m-%d %H:%M:%S")

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
                os.path.splitext(filename)[0]
                + f'_continued_page_{start}'
                + output_format
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
            sources.attrs['scope_code_version'] = code_version
            sources.attrs['fritz_download_dateTime_utc'] = start_dt

            if output_format == '.csv':
                sources.to_csv(filepath, index=False)
            elif output_format == '.h5':
                write_hdf(sources, filepath)
            else:
                write_parquet(sources, filepath)

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
                output_format,
            )
            return merged_sources

    else:
        # read in CSV, HDF5 or parquet file
        if file.endswith('.csv'):
            sources = pd.read_csv(file)
        elif file.endswith('.h5'):
            sources = read_hdf(file)
        elif file.endswith('.parquet'):
            sources = read_parquet(file)
        else:
            raise TypeError('Input file must be h5, csv or parquet format.')

        if start != 0:
            # continue from checkpoint
            sources = sources[start:]
            filename = (
                os.path.splitext(filename)[0]
                + f'_continued_index_{start}'
                + output_format
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
                        existing_attrs = sources.attrs
                        sources_chkpt = pd.merge(
                            sources, pd.json_normalize(dct_list), on='obj_id'
                        )
                        sources_chkpt.attrs['scope_code_version'] = code_version
                        sources_chkpt.attrs['fritz_download_dateTime_utc'] = start_dt
                        sources.attrs = {**sources.attrs, **existing_attrs}

                        if output_format == '.csv':
                            sources_chkpt.to_csv(filepath, index=False)
                        elif output_format == '.h5':
                            write_hdf(sources_chkpt, filepath)
                        else:
                            write_parquet(sources_chkpt, filepath)

                        print(f'Saved checkpoint at index {index}.')

                else:
                    warnings.warn(f'Unable to find source {index} on Fritz.')

            # final save
            existing_attrs = sources.attrs
            sources = pd.merge(sources, pd.json_normalize(dct_list), on='obj_id')
            sources.attrs['scope_code_version'] = code_version
            sources.attrs['fritz_download_dateTime_utc'] = start_dt
            sources.attrs = {**sources.attrs, **existing_attrs}

            print('Saving all sources.')
            if output_format == '.csv':
                sources.to_csv(filepath, index=False)
            elif output_format == '.h5':
                write_hdf(sources, filepath)
            else:
                write_parquet(sources, filepath)

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
                output_format,
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
        help="Name of directory to save downloaded file",
    )

    parser.add_argument(
        "-output_filename",
        type=str,
        default='merged_classifications_features',
        help="Name of output file containing merged classifications and features",
    )

    parser.add_argument(
        "-output_format",
        type=str,
        default='.parquet',
        help="Format of output file: .parquet, .h5 or .csv",
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
        args.output_format,
    )

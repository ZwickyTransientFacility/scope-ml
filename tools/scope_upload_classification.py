#!/usr/bin/env python
import argparse
import json as JSON
import pandas as pd
from penquins import Kowalski
from scope.fritz import save_newsource, api
from scope.utils import read_hdf, read_parquet, write_hdf, write_parquet
import math
import warnings
import yaml
import pathlib
from tools import scope_manage_annotation
from datetime import datetime
import numpy as np
import os

UPLOAD_BATCHSIZE = 10
OBJ_ID_BATCHSIZE = 10

config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)

# use tokens specified as env vars (if exist)
kowalski_token_env = os.environ.get("KOWALSKI_INSTANCE_TOKEN")
gloria_token_env = os.environ.get("GLORIA_INSTANCE_TOKEN")
melman_token_env = os.environ.get("MELMAN_INSTANCE_TOKEN")

# Set up Kowalski instance connection
if kowalski_token_env is not None:
    config["kowalski"]["hosts"]["kowalski"]["token"] = kowalski_token_env
if gloria_token_env is not None:
    config["kowalski"]["hosts"]["gloria"]["token"] = gloria_token_env
if melman_token_env is not None:
    config["kowalski"]["hosts"]["melman"]["token"] = melman_token_env

timeout = config['kowalski']['timeout']

hosts = [
    x
    for x in config['kowalski']['hosts']
    if config['kowalski']['hosts'][x]['token'] is not None
]
instances = {
    host: {
        'protocol': config['kowalski']['protocol'],
        'port': config['kowalski']['port'],
        'host': f'{host}.caltech.edu',
        'token': config['kowalski']['hosts'][host]['token'],
    }
    for host in hosts
}

kowalski_instances = Kowalski(timeout=timeout, instances=instances)


def upload_classification(
    file: str,
    kowalski_instances: Kowalski,
    group_ids: list,
    classification: list,
    taxonomy_map: str,
    comment: str,
    start: int,
    stop: int,
    classification_origin: str = 'SCoPe',
    post_survey_id: bool = False,
    survey_id_origin: str = 'SCoPe_xmatch',
    skip_phot: bool = False,
    p_threshold: float = 0.0,
    match_ids: bool = False,
    use_existing_obj_id: bool = False,
    post_upvote: bool = False,
    check_labelled_box: bool = False,
    write_obj_id: bool = False,
    result_dir: str = 'fritzUpload',
    result_filetag: str = 'fritzUpload',
    result_format: str = 'parquet',
    replace_classifications: bool = False,
    radius_arcsec: float = 2.0,
    no_ml: bool = False,
):
    """
    Upload labels to Fritz
    :param file: path to .csv, .h5 or .parquet file containing labels (str)
    :param kowalski_instances: authenticated kowalski instances (penquins.Kowalski)
    :param group_ids: list of group ids on Fritz for upload target location [int, int, ...]
    :param classification: list of classifications [str, str, ...]
    :param taxonomy_map: if classification is ['read'], path to JSON file containing taxonomy mapping (str)
    :param comment: single comment to post (str)
    :param start: index in CSV file to start upload (int)
    :param stop: index in CSV file to stop upload (inclusive) (int)
    :param classification_origin: origin of classifications (str)
    :post_survey_id: if True, post survey_id from input dataset (bool)
    :skip_phot: if True, only upload groups and classifications (no photometry) (bool)
    :p_threshold: classification probabilties must be >= this number to post (float)
    :match_ids: if True, match ZTF source ids when searching existing sources (bool)
    :use_existing_obj_id: if True, source obj_id from input dataset (bool)
    :post_upvote: if True, post upvote to new classifications (bool)
    :check_labelled_box: if True, check labelled box for source (bool)
    :write_obj_id: if True, write each obj_id to copy of input file (bool)
    :result_dir: directory to write results from upload (str)
    :result_filetag: tag to append to input filename after upload (str)
    :result_format: format of resulting uploaded file (str)
    :replace_classifications: if True, delete each object's existing classifications before posting new ones (bool)
    :radius_arcsec: photometry search radius for uploaded sources (float)
    :no_ml: if True, posted classifications are not noted to originate from an ML classifier (bool)
    """

    # read in file to csv
    if file.endswith('.csv'):
        all_sources = pd.read_csv(file)
    elif file.endswith('.h5'):
        all_sources = read_hdf(file)
    elif file.endswith('.parquet'):
        all_sources = read_parquet(file)
    else:
        raise TypeError('Input file must be csv, h5 or parquet format.')

    if len(all_sources) == 0:
        warnings.warn("No sources to upload.")
        return

    columns = all_sources.columns
    sources = all_sources.copy()

    if start is not None:
        sources = sources.loc[start:]
    else:
        start = sources.index[0]
    if stop is not None:
        sources = sources.loc[:stop]
    else:
        stop = sources.index[-1]

    # for classification "read" mode, load taxonomy map
    read_classes = False
    if classification is not None:
        if (
            (classification[0] == "read")
            | (classification[0] == 'Read')
            | (classification[0] == 'READ')
        ):
            read_classes = True

        with open(taxonomy_map, 'r') as f:
            tax_map = JSON.load(f)

        classes = [key for key in tax_map.keys()]  # define list of columns to examine

    dict_list = []
    obj_id_dict = {}
    for index, row in sources.iterrows():
        probs = {}
        cls_list = []
        tax_dict = {}
        existing_classes = []

        if classification is not None:
            if not read_classes:
                # Allow subset of all mapper classes to be specified for upload
                classes = list(set.intersection(set(classification), set(classes)))
            row_classes = row[classes]  # limit current row to specified columns
            if p_threshold > 0.0:
                threshold_keys = row_classes.keys()[
                    row_classes >= p_threshold
                ]  # determine which dataset classifications are nonzero
            # Do not post classifications if p = 0
            else:
                threshold_keys = row_classes.keys()[row_classes > 0]

            for val in threshold_keys:
                cls = tax_map[val]['fritz_label']
                tax_id = tax_map[val]['taxonomy_id']
                if cls != 'None':  # if Fritz taxonomy value exists, add to class list
                    probs[cls] = row[val]
                    cls_list += [cls]
                    tax_dict[cls] = tax_id

        ra, dec = float(row.ra), float(row.dec)

        # Check for and assign period
        period = None  # default
        if 'period' in columns:
            period = float(row.period)
            if math.isnan(period):
                period = None
                warnings.warn('period is NaN - skipping period upload.')
        else:
            warnings.warn('period column is missing - skipping period upload.')

        existing_source = []
        n_missing_groups = 0
        src_dict = {}
        if post_survey_id:
            survey_id = row['survey_id']

        if ('obj_id' in columns) & (use_existing_obj_id):
            obj_id = row.obj_id
            response = api("GET", f"/api/sources/{obj_id}")
            data = response.json().get('data')
            existing_source = data
            if len(data) > 0:
                src_dict[obj_id] = data
        else:
            obj_id = None
            # get object id
            response = api(
                "GET", f"/api/sources?&ra={ra}&dec={dec}&radius={radius_arcsec/3600}"
            )
            data = response.json().get('data')

            create_time_dict = {}

            if data["totalMatches"] > 0:
                # get most recent ZTFJ source
                id_found = 0
                for src in data['sources']:
                    if id_found:
                        break
                    src_id = src['id']
                    src_dict[src_id] = src

                    if match_ids:
                        print("Attempting to match with survey_id annotation.")
                        annotations = src['annotations']
                        for annot in annotations:
                            if annot['origin'] == survey_id_origin:
                                src_survey_ids = [x for x in annot['data'].values()]
                                if survey_id in src_survey_ids:
                                    id_found = 1
                                    break
                    if not id_found:
                        print('Searching most recent ZTFJ source IDs...')
                        create_time = src['created_at']
                        dt_create_time = datetime.strptime(
                            create_time, '%Y-%m-%dT%H:%M:%S.%f'
                        )
                        create_time_dict[dt_create_time] = src_id

                if id_found:
                    existing_source = src_dict[src_id]
                    obj_id = src_id
                else:
                    create_time_list = [x for x in create_time_dict.keys()]
                    create_time_list.sort(reverse=True)

                    for t in create_time_list:
                        src_id = create_time_dict[t]
                        if src_id[:4] == 'ZTFJ':
                            # Treat source as new if not posted to all user-specified groups
                            posted_groups = [
                                x['id'] for x in src_dict[src_id]['groups']
                            ]
                            for gid in group_ids:
                                if gid not in posted_groups:
                                    n_missing_groups += 1
                            existing_source = src_dict[src_id]
                            obj_id = src_id
                            if n_missing_groups > 0:
                                print(
                                    'Source exists but is not posted to all specified groups. Treating as a new source for non-member groups.'
                                )
                            break

        print(f"object {index} id:", obj_id)

        # save_newsource can only be skipped if source exists in all specified groups
        if (len(existing_source) == 0) | (not skip_phot) | (n_missing_groups > 0):
            if ((len(existing_source) == 0) | (n_missing_groups > 0)) & (skip_phot):
                warnings.warn('Cannot skip new source - saving.')
            post_source = len(existing_source) == 0
            obj_id = save_newsource(
                kowalski_instances,
                group_ids,
                ra,
                dec,
                obj_id=obj_id,
                period=period,
                return_id=True,
                radius=radius_arcsec,
                post_source=post_source,
                skip_phot=skip_phot,
            )

        data_groups = []
        data_classes = []
        if obj_id is not None:

            # check which groups source is already in
            add_group_ids = group_ids.copy()
            if len(existing_source) > 0:
                data_groups = existing_source['groups']
                data_classes = existing_source['classifications']

            # remove existing groups from list of groups
            for entry in data_groups:
                existing_group_id = entry['id']
                if existing_group_id in add_group_ids:
                    add_group_ids.remove(existing_group_id)

            if len(add_group_ids) > 0:
                # save to new group_ids
                json = {"objId": obj_id, "inviteGroupIds": add_group_ids}
                response = api("POST", "/api/source_groups", json)

            # check for existing classifications and their groups
            class_group_dict = {}
            class_id_dict = {}
            not_in_group_count = 0
            check_group_ids = group_ids.copy()

            for entry in data_classes:
                c_key = entry['classification']
                c_id = [entry['id']]
                c_values = [x['id'] for x in entry['groups']]
                # If a classification has no groups, include it for replacement
                if c_values == []:
                    # Use -1 as placeholder for 'no group', since no existing group will have this id
                    c_values = [-1]
                    if -1 not in check_group_ids:
                        check_group_ids.append(-1)

                if len(list(set.intersection(set(check_group_ids), set(c_values)))) > 0:
                    if c_key not in class_group_dict.keys():
                        class_group_dict[c_key] = c_values
                        class_id_dict[c_key] = c_id
                    else:
                        class_group_dict[entry['classification']].extend(c_values)
                        class_id_dict[entry['classification']].extend(c_id)
                else:
                    not_in_group_count += 1

            for key in class_group_dict.keys():
                grp_ids = np.array(class_group_dict[key])
                unique_grp_ids = np.unique(grp_ids)
                class_group_dict[key] = unique_grp_ids.tolist()

            existing_classes = [k for k in class_group_dict.keys()]
            remaining_classes = existing_classes.copy()

            # If all existing classifications are in at least one of the user's specified groups
            # and replace_classifications is given, delete all classifications
            if (len(existing_classes) > 0) & (replace_classifications):
                if not_in_group_count == 0:
                    api(
                        'DELETE',
                        f'/api/sources/{obj_id}/classifications',
                        {'label': check_labelled_box},
                    )
                    existing_classes = []

                else:
                    # Otherwise, delete classifications that are within user-specified groups one-by-one
                    for exst_cls in existing_classes:
                        for class_id in class_id_dict[exst_cls]:
                            api(
                                'DELETE',
                                f'api/classification/{class_id}',
                                {'label': check_labelled_box},
                            )
                        remaining_classes.remove(exst_cls)
                    existing_classes = remaining_classes

            # allow classification assignment to be skipped
            if classification is not None:
                for cls in cls_list:
                    tax = tax_dict[cls]
                    prob = probs[cls]
                    json = {
                        "obj_id": obj_id,
                        "classification": cls,
                        "origin": classification_origin,
                        "taxonomy_id": tax,
                        "probability": prob,
                        "group_ids": group_ids,
                        "vote": post_upvote,
                        "label": check_labelled_box,
                        "ml": not no_ml,
                    }
                    if cls not in existing_classes:
                        # post all non-duplicate classifications
                        dict_list += [json]
                    else:
                        # Classification may exist, but not for intended groups.
                        groups_to_post = []
                        for g in group_ids:
                            if g not in class_group_dict[cls]:
                                groups_to_post += [g]
                        if len(groups_to_post) > 0:
                            dict_list += [json]

            if comment is not None:
                # get comment text
                response_comments = api("GET", f"/api/sources/{obj_id}/comments")
                data_comments = response_comments.json().get("data")

                # check for existing comments
                existing_comments = []
                for entry in data_comments:
                    existing_comments += [entry['text']]

                # post all non-duplicate comments
                if comment not in existing_comments:
                    json = {
                        "text": comment,
                    }
                    response = api("POST", f"/api/sources/{obj_id}/comments", json)

            # Post ZTF ID as annotation
            if post_survey_id:
                if obj_id not in [x for x in src_dict.keys()]:
                    scope_manage_annotation.manage_annotation(
                        'POST',
                        obj_id,
                        group_ids,
                        survey_id_origin,
                        'survey_id',
                        survey_id,
                    )
                else:
                    source_to_update = src_dict[obj_id]
                    existing_survey_ids = [
                        x['data']
                        for x in source_to_update['annotations']
                        if x['origin'] == survey_id_origin
                    ]
                    if len(existing_survey_ids) == 0:
                        scope_manage_annotation.manage_annotation(
                            'POST',
                            obj_id,
                            group_ids,
                            survey_id_origin,
                            'survey_id',
                            survey_id,
                        )
                    else:
                        n_existing_survey_ids = len(existing_survey_ids[0].keys())
                        if survey_id not in existing_survey_ids[0].values():
                            scope_manage_annotation.manage_annotation(
                                'UPDATE',
                                obj_id,
                                group_ids,
                                survey_id_origin,
                                f'survey_id_{n_existing_survey_ids+1}',
                                str(survey_id),
                            )

        else:
            obj_id = ''

        obj_id_dict[index] = obj_id

        # batch upload classifications
        if len(dict_list) != 0:
            if (((index - start) + 1) % UPLOAD_BATCHSIZE == 0) | (index == stop):
                print('uploading classifications...')
                json_classes = {'classifications': dict_list}
                api("POST", "/api/classification", json_classes)
                dict_list = []

        if write_obj_id:
            if (((index - start) + 1) % OBJ_ID_BATCHSIZE == 0) | (index == stop):
                # Check for appropriate result format
                result_file_extension = os.path.splitext(result_filetag)[-1]

                # If parquet, h5 or csv extension specified in result_filetag, use that format for saving
                result_format = (
                    result_format
                    if result_file_extension not in ['.parquet', '.h5', '.csv']
                    else result_file_extension
                )

                if result_format in [
                    '.parquet',
                    'parquet',
                    '.Parquet',
                    'Parquet',
                    '.parq',
                    'parq',
                    '.PARQUET',
                    'PARQUET',
                    '.PARQ',
                    'PARQ',
                ]:
                    result_format = '.parquet'
                    print('Using .parquet extension for saved files.')
                elif result_format in [
                    '.h5',
                    'h5',
                    '.H5',
                    'H5',
                    '.hdf5',
                    'hdf5',
                    '.HDF5',
                    'HDF5',
                ]:
                    result_format = '.h5'
                    print('Using .h5 extension for saved files.')
                elif result_format in ['.csv', 'csv', '.CSV', 'CSV']:
                    result_format = '.csv'
                    print('Using .csv extension for saved files.')
                else:
                    raise ValueError('result format must be parquet, hdf5 or csv.')

                # If user puts extension in filename, remove it for consistency
                if (
                    (result_filetag.endswith('.csv'))
                    | (result_filetag.endswith('.h5'))
                    | (result_filetag.endswith('.parquet'))
                ):
                    result_filetag = os.path.splitext(result_filetag)[0]

                outpath = os.path.join(os.path.dirname(__file__), result_dir)
                os.makedirs(outpath, exist_ok=True)

                filename = (
                    os.path.splitext(os.path.basename(file))[0]
                    + '_'
                    + result_filetag
                    + result_format
                )  # rename file
                filepath = os.path.join(outpath, filename)

                sources_to_write = sources.loc[[x for x in obj_id_dict.keys()]]
                sources_to_write['obj_id'] = [x for x in obj_id_dict.values()]

                if 'obj_id' not in all_sources.columns:
                    all_sources['obj_id'] = ''
                all_sources.loc[start:stop, 'obj_id'] = sources_to_write['obj_id']

                print(
                    'Saving obj_id for uploaded sources. If upload is incomplete, use this file for future uploads to continue filling the obj_id column.'
                )
                print()
                if result_format == '.csv':
                    all_sources.to_csv(filepath, index=False)
                elif result_format == '.h5':
                    write_hdf(all_sources, filepath)
                else:
                    write_parquet(all_sources, filepath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="dataset with .csv, .h5 or .parquet extension")
    parser.add_argument("--group_ids", type=int, nargs='+', help="list of group ids")

    # parser.add_argument("-classification", type=str, help="name of object class")
    parser.add_argument(
        "--classification", type=str, nargs='+', help="list of object classes"
    )
    parser.add_argument(
        "--taxonomy_map",
        type=str,
        help="JSON file mapping between origin labels and Fritz taxonomy",
    )
    parser.add_argument(
        "--comment",
        type=str,
        help="Post specified string to comments for sources in file",
    )
    parser.add_argument("--start", type=int, help="Zero-based index to start uploading")
    parser.add_argument(
        "--stop",
        type=int,
        help="Index to stop uploading (inclusive)",
    )
    parser.add_argument(
        "--classification_origin",
        type=str,
        default='SCoPe',
        help="origin of classifications",
    )
    parser.add_argument(
        "--skip_phot",
        type=bool,
        nargs='?',
        default=False,
        const=True,
        help="Skip photometry upload, only post groups and classifications.",
    )

    parser.add_argument(
        "--post_survey_id",
        action='store_true',
        help="If set, post survey_id from input dataset.",
    )

    parser.add_argument(
        "--survey_id_origin",
        type=str,
        default='SCoPe_xmatch',
        help="Annotation origin for survey ID",
    )

    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.0,
        help="Classification probability >= this number to upload",
    )

    parser.add_argument(
        "--match_ids",
        action='store_true',
        default=False,
        help="If set, match input and existing sources using ZTF source IDs.",
    )

    parser.add_argument(
        "--use_existing_obj_id",
        action='store_true',
        default=False,
        help="If set, source obj_id from input dataset.",
    )

    parser.add_argument(
        "--post_upvote",
        action='store_true',
        default=False,
        help="If set, post upvote to new classifications.",
    )

    parser.add_argument(
        "--check_labelled_box",
        action='store_true',
        default=False,
        help="If set, check 'labelled' box for source.",
    )

    parser.add_argument(
        "--write_obj_id",
        action='store_true',
        default=False,
        help="If set, write obj_ids corresponding to each uploaded source.",
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        default='fritzUpload',
        help="Directory to save upload results",
    )

    parser.add_argument(
        "--result_filetag",
        type=str,
        default='fritzUpload',
        help="Directory to save upload results",
    )

    parser.add_argument(
        "--result_format",
        type=str,
        default='parquet',
        help="Format of result file: parquet, h5 or csv",
    )

    parser.add_argument(
        "--replace_classifications",
        action='store_true',
        default=False,
        help="If set, delete each object's existing classifications before posting new ones.",
    )

    parser.add_argument(
        "--radius_arcsec",
        type=float,
        default=2.0,
        help="Photometry search radius for uploaded sources",
    )

    parser.add_argument(
        "--no_ml",
        action='store_true',
        default=False,
        help="If set, posted classifications are not noted to originate from an ML classifier.",
    )

    args = parser.parse_args()

    # upload classification objects
    upload_classification(
        args.file,
        kowalski_instances,
        args.group_ids,
        args.classification,
        args.taxonomy_map,
        args.comment,
        args.start,
        args.stop,
        args.classification_origin,
        args.post_survey_id,
        args.survey_id_origin,
        args.skip_phot,
        args.p_threshold,
        args.match_ids,
        args.use_existing_obj_id,
        args.post_upvote,
        args.check_labelled_box,
        args.write_obj_id,
        args.result_dir,
        args.result_filetag,
        args.result_format,
        args.replace_classifications,
        args.radius_arcsec,
        args.no_ml,
    )

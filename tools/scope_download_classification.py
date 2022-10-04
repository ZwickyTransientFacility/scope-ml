#!/usr/bin/env python
import argparse
import pandas as pd
from scope.fritz import api
import warnings
import numpy as np

# from time import sleep

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


def download_classification(file: str, group_ids: list, token: str, start: int):
    """
    Download labels from Fritz
    :param file: CSV file containing obj_id column or "parse" to query by group ids (str)
    :param group_ids: target group ids on Fritz for download (list)
    :param token: Fritz token (str)
    """

    dict_list = []

    filename = file.removesuffix('.csv') + '_fritzDownload' + '.csv'  # rename file

    if file in ["parse", 'Parse', 'PARSE']:
        if group_ids is None:
            raise ValueError('Specify group_ids to query Fritz.')
        response = api(
            "GET",
            "/api/sources",
            token,
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
                token,
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
            sources.to_csv(filename, index=False)
            print(f'Saved page {pageNum}.')

        return sources

    else:
        # read in CSV file
        sources = pd.read_csv(file)
        if start != 0:
            # continue from checkpoint
            sources = sources[start:]
            filename = (
                filename.removesuffix('.csv') + f'_continued_index_{start}' + '.csv'
            )

        columns = sources.columns

        # create new empty columns
        sources["classification"] = None
        sources["probability"] = None
        sources["period_origin"] = None
        sources["period"] = None
        sources["ztf_id_origin"] = None
        sources["ztf_id"] = None
        # add obj_id column if not passed in
        if 'obj_id' not in columns:
            sources["obj_id"] = None
            search_by_obj_id = False
        else:
            search_by_obj_id = True

        for index, row in sources.iterrows():
            # query objects, starting with obj_id
            data = []
            if search_by_obj_id:
                obj_id = row.obj_id
                response = api("GET", '/api/sources/%s' % obj_id, token)
                # sleep(0.9)
                data = response.json().get("data")
                if len(data) == 0:
                    warnings.warn('No results from obj_id search - querying by ra/dec.')
                else:
                    src = data

            # continue with coordinate search if obj_id unsuccsessful
            if len(data) == 0:
                if ('ra' in columns) & ('dec' in columns):
                    # query by ra/dec to get object id
                    ra, dec = row.ra, row.dec
                    response = api(
                        "GET", f"/api/sources?&ra={ra}&dec={dec}&radius={2/3600}", token
                    )
                    # sleep(0.9)
                    data = response.json().get("data")
                    obj_id = None
                    if data["totalMatches"] > 0:
                        src = data["sources"][0]
                        obj_id = src["id"]
                        sources.at[index, 'obj_id'] = obj_id
                else:
                    raise KeyError(
                        'Attemped to search by coordinates, but unable to find ra and dec columns.'
                    )

            print(f"object {index} id:", obj_id)

            # if successful search, get and save labels/probabilities/period annotations to sources dataframe
            if obj_id is not None:
                (
                    id,
                    ra,
                    dec,
                    cls_list,
                    prb_list,
                    origin_list,
                    period_list,
                    id_origin_list,
                    id_list,
                ) = organize_source_data(src)

                # store to new columns
                sources.at[index, 'ra'] = ra
                sources.at[index, 'dec'] = dec
                sources.at[index, 'classification'] = cls_list
                sources.at[index, 'probability'] = prb_list
                sources.at[index, 'period_origin'] = origin_list
                sources.at[index, 'period'] = period_list
                sources.at[index, 'ztf_id_origin'] = origin_list
                sources.at[index, 'ztf_id'] = period_list

                # occasional checkpoint at specified number of sources
                if (index + 1) % CHECKPOINT_NUM == 0:
                    sources.to_csv(filename, index=False)
                    print(f'Saved checkpoint at index {index}.')

            else:
                warnings.warn(f'Unable to find source {index} on Fritz.')

            # final save
            sources.to_csv(filename, index=False)

        return sources


if __name__ == "__main__":

    # pass Fritz token as command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", help="dataset")
    parser.add_argument("-group_ids", type=int, nargs='+', help="list of group ids")
    parser.add_argument(
        "-token",
        type=str,
        help="put your Fritz token here. You can get it from your Fritz profile page",
    )
    parser.add_argument(
        "-start", type=int, default=0, help="start page/index for continued download"
    )
    args = parser.parse_args()

    # download object classifications in the file
    download_classification(args.file, args.group_ids, args.token, args.start)

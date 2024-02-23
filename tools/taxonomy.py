#!/usr/bin/env python
import requests
from typing import List, Optional
import yaml
from inspect import ismodule
from scope.utils import parse_load_config
import argparse


config = parse_load_config()


def fritz_api(method: str, endpoint: str, data: Optional[dict] = None):
    headers = {"Authorization": f"token {config['fritz']['token']}"}
    response = requests.request(method, endpoint, json=data, headers=headers)
    return response


def post_taxonomy(
    taxonomy,
    group_ids: Optional[List[int]] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    provenance: Optional[str] = None,
):
    """Post taxonomy to Fritz
       NOTE: token in config.yaml must have 'Post taxonomy' permission

    $ post-taxonomy \
      --taxonomy phenomenological.yaml \
      --group_ids 1444 \
      --name "Scope Phenomenological Taxonomy" \
      --version 1.2.0  \
      --provenance https://github.com/bfhealy/scope-phenomenology.git

    Within python:
    taxonomy.post(scope_phenom, group_ids=[1, 2])

    :param taxonomy: path to yaml file with taxonomy in tdtax format, or imported taxonomy module
    :param group_ids: ids of groups on Fritz to post taxonomy to.
                      if None, will post to all user (token) groups (int or list)
    :param name: name of input taxonomy (str)
    :param version: version of input taxonomy (str)
    :param provenance: URL hosting input taxonomy (str)

    :return:
    """

    # Read .yaml file and check other arguments
    if isinstance(taxonomy, str):
        with open(taxonomy) as taxonomy_yaml:
            tax = yaml.load(taxonomy_yaml, Loader=yaml.FullLoader)
        if (name is None) | (version is None) | (provenance is None):
            raise ValueError('Must specify name, version and provenance.')

    # Get attributes from imported taxonomy module
    elif ismodule(taxonomy):
        tax = taxonomy.taxonomy
        name = taxonomy.name
        version = taxonomy.__version__
        provenance = taxonomy.provenance

    else:
        raise TypeError('--taxonomy must be string or module.')

    tax_obj = {
        'name': name,
        'provenance': provenance,
        'version': version,
        'hierarchy': tax,
    }

    if group_ids is not None:
        if not hasattr(group_ids, "__iter__"):
            group_ids = (group_ids,)
        tax_obj["group_ids"] = list(group_ids)

    response = fritz_api(
        "POST",
        f"{config['fritz']['protocol']}://{config['fritz']['host']}/api/taxonomy",
        tax_obj,
    )
    print(response.json())

    response_json = response.json()
    if response_json['status'] == 'success':
        tax_id = response_json['data']['taxonomy_id']
        print(f'Posted taxonomy (ID = {tax_id}.)')
    else:
        message = response_json['message']
        print(f'Did not post taxonomy - message: {message}.')


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--taxonomy",
        type=str,
        default="phenomenological.yaml",
        help="path to yaml file with taxonomy in tdtax format, or imported taxonomy module",
    )
    parser.add_argument(
        "--group-ids",
        type=int,
        nargs='+',
        help="ids of groups on Fritz to post taxonomy to (all if not specified).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Scope Phenomenological Taxonomy",
        help="name of input taxonomy",
    )
    parser.add_argument(
        "--version",
        type=str,
        default='1.0.0',
        help="version of input taxonomy",
    )
    parser.add_argument(
        "--provenance",
        type=str,
        default="github",
        help="URL hosting input taxonomy",
    )

    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    post_taxonomy(**vars(args))

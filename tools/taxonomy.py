#!/usr/bin/env python
import fire
import pathlib
import requests
from typing import List, Optional
import yaml
from inspect import ismodule


config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)


def fritz_api(method: str, endpoint: str, data: Optional[dict] = None):
    headers = {"Authorization": f"token {config['fritz']['token']}"}
    response = requests.request(method, endpoint, json=data, headers=headers)
    return response


def post(
    taxonomy,
    group_ids: Optional[List[int]] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    provenance: Optional[str] = None,
):
    """Post taxonomy to Fritz

    $ ./taxonomy.py \
      --taxonomy='phenomenological.yaml' \
      --group_ids=1444 \
      --name='Scope Phenomenological Taxonomy' \
      --version='1.2.0' \
      --provenance='https://github.com/bfhealy/scope-phenomenology.git'

    $ ./taxonomy.py \
      --taxonomy=scope_phenom \
      --group_ids=339

    :param taxonomy: path to yaml file with taxonomy in tdtax format, or imported taxonomy module
    :param group_ids: ids of groups on Fritz to post taxonomy to.
                      if None, will post to all user (token) groups (int or list)
    :param name: name of input taxonomy (str)
    :param version: version of input taxonomy (str)
    :param provenance: URL hosting input taxonomy (str)
    :return:
    """

    # Read .yaml file and check other arguments
    if type(taxonomy) == str:
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


if __name__ == "__main__":
    fire.Fire(post)

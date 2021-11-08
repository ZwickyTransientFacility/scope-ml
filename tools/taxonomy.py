import fire
import pathlib
import requests
from typing import List, Optional
import yaml


config_path = pathlib.Path(__file__).parent.parent.absolute() / "config.yaml"
with open(config_path) as config_yaml:
    config = yaml.load(config_yaml, Loader=yaml.FullLoader)


def fritz_api(method: str, endpoint: str, data: Optional[dict]):
    headers = {"Authorization": f"token {config['fritz']['token']}"}
    response = requests.request(method, endpoint, json=data, headers=headers)
    return response


def post(taxonomy: str, group_ids: Optional[List[int]] = None):
    """Post taxonomy to Fritz

    $ python tools/taxonomy.py \
      --taxonomy=tools/scope_taxonomy.yaml \
      --group_id=339

    :param taxonomy: path to yaml file with taxonomy in tdtax format
    :param group_ids: ids of groups on Fritz to post taxonomy to.
                      if None, will post to all user (token) groups
    :return:
    """
    with open(taxonomy) as taxonomy_yaml:
        tax = yaml.load(taxonomy_yaml, Loader=yaml.FullLoader)

    if group_ids is not None:
        if not hasattr(group_ids, "__iter__"):
            group_ids = (group_ids,)
        tax["group_ids"] = list(group_ids)

    response = fritz_api(
        "POST",
        f"{config['fritz']['protocol']}://{config['fritz']['host']}/api/taxonomy",
        tax,
    )
    print(response.json())


if __name__ == "__main__":
    fire.Fire(post)

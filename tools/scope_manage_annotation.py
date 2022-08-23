#!/usr/bin/env python
import argparse
from scope.fritz import api


def manage_annotation(action, obj_id, group_ids, origin, key, value, token):

    # Forgiving definitions of actions
    if action in ['update', 'Update', 'UPDATE']:
        action = 'update'
    elif action in ['delete', 'Delete', 'DELETE']:
        action = 'delete'
    elif action in ['post', 'Post', 'POST']:
        action = 'post'

    # Check for dict if performing update or post actions
    if (value is None) & ((action == 'update') | (action == 'post')):
        print('Error: please specify annotation value for -data to update or post.')
        return

    if (action == 'update') | (action == 'delete'):
        # get all annotations for object
        response = api("GET", '/api/sources/%s/annotations' % obj_id, token).json()
        data = response.get('data')

        # loop through annotations, checking for match with input key and origin
        for entry in data:
            annot_id = str(entry['id'])
            annot_origin = entry['origin']
            annot_data = entry['data']

            annot_name = [x for x in annot_data.keys()][0]
            annot_value = [x for x in annot_data.values()][0]

            # if match is found, perform action
            if (key == annot_name) & (origin == annot_origin):
                if action == 'update':
                    json = {"data": {key: value}, "origin": origin, "obj_id": annot_id}
                    response = api(
                        "PUT",
                        '/api/sources/%s/annotations/%s' % (obj_id, annot_id),
                        token,
                        json,
                    )
                    if response.status_code == 200:
                        print(
                            'Updated annotation %s (%s = %s to %s) for %s'
                            % (annot_origin, annot_name, annot_value, value, obj_id)
                        )
                    return

                elif action == 'delete':
                    response = api(
                        "DELETE",
                        '/api/sources/%s/annotations/%s' % (obj_id, annot_id),
                        token,
                    )
                    if response.status_code == 200:
                        print(
                            'Deleted annotation %s (%s = %s) for %s'
                            % (annot_origin, annot_name, annot_value, obj_id)
                        )
                    return

        print("Could not find key/origin combination in existing annotations.")

    # if posting new annotation, skip search for exisiting ones
    elif action == 'post':
        json = {"origin": origin, "data": {key: value}, "group_ids": group_ids}
        response = api("POST", '/api/sources/%s/annotations' % obj_id, token, json)
        if response.status_code == 200:
            print(
                'Posted annotation %s (%s = %s) for %s' % (origin, key, value, obj_id)
            )

    else:
        print("Error: Please specify action as one of 'post', 'update', or 'delete'.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-action", help="post, update, or delete annotation")
    parser.add_argument("-obj_id", help="Fritz object id")
    parser.add_argument("-group_ids", type=int, nargs='+', help="list of group ids")
    parser.add_argument("-origin", type=str, help="name of annotation origin")
    parser.add_argument("-key", help="annotation key")
    parser.add_argument("-value", type=float, help="annotation value")
    parser.add_argument(
        "-token",
        type=str,
        help="put your Fritz token here. You can get it from your Fritz profile page",
    )

    args = parser.parse_args()

    manage_annotation(
        args.action,
        args.obj_id,
        args.group_ids,
        args.origin,
        args.key,
        args.value,
        args.token,
    )

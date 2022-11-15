import argparse
import os
import json
import pickle
import numpy as np
import geopandas as gpd
from sklearn.metrics import pairwise_distances
from collections import defaultdict

from sklearn.neighbors import BallTree


def get_nearest(src_points, candidates, k_neighbors=10, remove_first=True):
    """Find nearest neighbors for all source points from a set of candidate points"""
    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric="euclidean")
    # Find closest points and distances
    distances, indices = tree.query(
        src_points, k=k_neighbors + int(remove_first)
    )
    # Return indices and distances
    return (indices[:, remove_first:], distances[:, remove_first:])


def get_ordered_unique(arr):
    set_new, elems_new = set(), []
    for elem in arr:
        if elem not in set_new:
            set_new.add(elem)
            elems_new.append(elem)
    return elems_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_path",
        default="data_collection/example_pois.geojson",
        type=str
    )
    parser.add_argument("-p", "--positive_samples", default=10, type=int)
    parser.add_argument(
        "-o",
        "--out_path",
        default="data_collection/example_poi_data",
        type=str
    )
    args = parser.parse_args()

    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    nr_neighbors = args.positive_samples

    # LOAD data
    poi = gpd.read_file(args.data_path)
    mapping_prev_ids = {
        i: int(old_id)
        for i, old_id in enumerate(poi["id"].values)
    }
    with open(os.path.join(out_path, "poi_id_mapping.json"), "w") as outfile:
        json.dump(mapping_prev_ids, outfile)
    print("Saved mapping from old IDs to new IDs")
    poi["id"] = np.arange(len(poi))
    poi.set_index("id", inplace=True)

    # PART 1: POI types
    # add the main categories:
    poi_type_cols = [col for col in poi if col.startswith("poi_type_")]
    all_types = set()
    for poi_col in poi_type_cols:
        for elem in poi[poi_col].unique():
            all_types.add(elem)
    poi_id_mapping = {elem: i for i, elem in enumerate(list(all_types))}
    # reversed
    id_poi_mapping = {str(i): elem for elem, i in poi_id_mapping.items()}

    # SAVE the poi types
    with open(os.path.join(out_path, "poi_type.json"), "w") as outfile:
        json.dump(id_poi_mapping, outfile)
    print("Saved POI types")

    # PART 2: POI list with categories
    # update table
    for col in poi_type_cols:
        # transfer into numerical category IDs
        poi[col] = poi[col].map(poi_id_mapping)
    # train test splot
    rand_perm = np.random.permutation(len(poi))
    train_cutoff = int(len(poi) * 0.8)
    val_cutoff = int(len(poi) * 0.9)
    split_label_arr = np.array(["training"
                                for _ in range(len(poi))]).astype(str)
    split_label_arr[rand_perm[train_cutoff:val_cutoff]] = "validation"
    split_label_arr[rand_perm[val_cutoff:]] = "test"
    poi["split"] = split_label_arr
    poi.loc[poi["split"] == "validati", "split"] = "validation"
    # convert table into tuple
    my_poi_data = []
    for elem_id, row in poi.iterrows():
        this_tuple = (
            elem_id,
            (row["geometry"].x, row["geometry"].y),
            tuple([row[poi_type] for poi_type in poi_type_cols]),
            row["split"],
        )
        my_poi_data.append(this_tuple)
    number_of_pois = len(id_poi_mapping)

    # Save the poi data with the categories
    with open(os.path.join(out_path, "pointset.pkl"), "wb") as outfile:
        pickle.dump((number_of_pois, my_poi_data), outfile)
    print("Saved POI-label data")

    # PART 3: sample the spatially closest
    coord_arr = np.swapaxes(
        np.vstack([poi["geometry"].x.values, poi["geometry"].y.values]), 1, 0
    )
    closest, distance_of_closest = get_nearest(
        coord_arr, coord_arr, k_neighbors=nr_neighbors
    )
    print("Finished positive sampling")

    # convert index
    poi_id_list = list(poi.index)
    poi_id_array = np.array(poi_id_list)
    poi_id_set = set(poi_id_list)

    # Negative sampling:
    all_tuples = []
    for counter, positive_sampled_index in enumerate(closest):
        elem_id = poi_id_list[counter]
        positive_sampled = poi_id_array[positive_sampled_index]
        leftover = list(poi_id_set - set([elem_id] + list(positive_sampled)))
        negative_sampled = list(np.random.choice(leftover, nr_neighbors))

        mode = poi.loc[elem_id, "split"]
        all_tuples.append(
            (
                elem_id, tuple(positive_sampled), mode, negative_sampled,
                distance_of_closest[counter]
            )
        )
    print("Finisher negative sampling")

    for mode in ["training", "validation", "test"]:
        out_tuple = [
            the_tuple for the_tuple in all_tuples if the_tuple[2] == mode
        ]
        with open(
            os.path.join(out_path, f"neighborgraphs_{mode}.pkl"), "wb"
        ) as outfile:
            pickle.dump(out_tuple, outfile)
        print("Saved graph data", mode)

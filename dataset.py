import argparse
import json
import os
import random

import numpy as np
from tqdm import tqdm


def dataset_fn():
    def parse_line(ndjson_line):
        """Parse an ndjson line and return ink (as np array) and classname."""
        sample = json.loads(ndjson_line)
        class_name = sample["word"]
        if not class_name:
            print("Empty classname")
            return None, None
        inkarray = sample["drawing"]
        stroke_lengths = [len(stroke[0]) for stroke in inkarray]
        total_points = sum(stroke_lengths)
        np_ink = np.zeros((total_points, 3), dtype=np.float32)
        current_t = 0
        if not inkarray:
            print("Empty inkarray")
            return None, None
        for stroke in inkarray:
            if len(stroke[0]) != len(stroke[1]):
                print("Inconsistent number of x and y coordinates.")
                return None, None
            for i in [0, 1]:
                np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
            current_t += len(stroke[0])
            np_ink[current_t - 1, 2] = 1  # stroke_end
        # Preprocessing.
        # 1. Size normalization.
        lower = np.min(np_ink[:, 0:2], axis=0)
        upper = np.max(np_ink[:, 0:2], axis=0)
        scale = upper - lower
        scale[scale == 0] = 1
        np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
        # 2. Compute deltas.
        np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
        np_ink = np_ink[1:, :]
        return np_ink, class_name

    def convert_data(trainingdata_dir, observations_per_class, output_file, classnames=[], output_shards=10, offset=0):
        def _pick_output_shard():
            return random.randint(0, output_shards - 1)

        file_handles = []
        for filename in sorted(os.listdir(trainingdata_dir)):
            if not filename.endswith(".ndjson"):
                print("Skipping", filename)
                continue
            print("Find", filename)
            file_handles.append(
                open(os.path.join(trainingdata_dir, filename), 'r'))
            if offset:  # Fast forward all files to skip the offset.
                for i in range(offset):
                    file_handles[-1].readline()

        writers = []
        for i in range(output_shards):
            writers.append(open("%s-%05i-of-%05i" %
                                (output_file, i, output_shards), 'w'))

        reading_order = list(range(len(file_handles))) * observations_per_class
        random.shuffle(reading_order)
        print("Reading order length:", len(reading_order))

        for c in tqdm(reading_order):
            line = file_handles[c].readline()
            ink = None
            while ink is None:
                ink, class_name = parse_line(line)
                if ink is None:
                    print("Couldn't parse ink from '" + line + "'.")
            if class_name not in classnames:
                classnames.append(class_name)
            features = {}
            features["class_index"] = classnames.index(class_name)
            features["ink"] = ink.flatten().tolist()
            features["shape"] = ink.shape
            writers[_pick_output_shard()].write(json.dumps(features) + "\n")

        for w in writers:
            w.close()
        for f in file_handles:
            f.close()
        # Write the class list.
        with open(output_file + ".classes", "w") as f:
            for class_name in classnames:
                f.write(class_name + "\n")
        return classnames

    print("Processing training dataset...")
    classnames = convert_data(
        FLAGS.dataset_path,
        FLAGS.num_train,
        os.path.join(FLAGS.original_data_path, FLAGS.training_data),
        classnames=[],
        output_shards=10,
        offset=0)

    print("Processing evaluation dataset...")
    convert_data(
        FLAGS.dataset_path,
        FLAGS.num_eval,
        os.path.join(FLAGS.original_data_path, FLAGS.eval_data),
        classnames=classnames,
        output_shards=10,
        offset=FLAGS.offset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./dataset/",
        help="Path to store preprocessed dataset"
    )
    parser.add_argument(
        "--original_data_path",
        type=str,
        default="./data/",
        help="Path to store original data"
    )
    parser.add_argument(
        "--training_data",
        type=str,
        default="training.json",
        help="Filename to store training data")
    parser.add_argument(
        "--eval_data",
        type=str,
        default="eval.json",
        help="Filename to store evaluation data")
    parser.add_argument(
        "--num_train",
        type=int,
        default=500,
        help="Number of selections for each class in training")
    parser.add_argument(
        "--num_eval",
        type=int,
        default=150,
        help="Number of selections for each class in evaluation")
    parser.add_argument(
        "--offset",
        type=int,
        default=10000,
        help="Skipping count for evaluation dataset")

    FLAGS, _ = parser.parse_known_args()
    dataset_fn()

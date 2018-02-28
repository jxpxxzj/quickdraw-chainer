# coding: utf-8

import argparse
import json

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, optimizers, training
from chainer.datasets import TupleDataset
from chainer.training import extensions

from model import Classifier, Network
from PadIterator import PadIterator


def _parse_tfexample_fn(example_proto, mode="TRAIN"):
    parsed_features = json.loads(example_proto)

    if mode != "PREDICT":
        label = parsed_features["class_index"]

    parsed_features["ink"] = np.array(
        parsed_features["ink"], dtype="float32").reshape(parsed_features["shape"])
    parsed_features["shape"] = np.array(parsed_features["shape"])
    return parsed_features, label


def load_example(output_file, output_shards):
    out_dict = {}
    out_dict["shapes"] = []
    out_dict["inks"] = []
    out_dict["labels"] = []
    readers = []
    for i in range(output_shards):
        readers.append(open("%s-%05i-of-%05i" %
                            (output_file, i, output_shards), 'r'))
    reading_order = range(len(readers))

    for c in reading_order:
        lines = readers[c].readlines()
        for line in lines:
            parsed_features, label = _parse_tfexample_fn(line)
            if parsed_features["shape"][0] == 1:
                print(parsed_features["ink"], "was ignored.")
                continue
            out_dict["shapes"].append(parsed_features["shape"])
            out_dict["inks"].append(np.array(parsed_features["ink"]))
            out_dict["labels"].append(label)
    return out_dict


def get_num_classes(classes_file):
    filename = classes_file + ".classes"
    lines = open(filename, 'r').readlines()
    return len(lines)


def get_tuple_dataset(data):
    slice_array = data["inks"]
    label_array = data["labels"]
    length_array = [c[0] for c in data["shapes"]]
    zipped = list(zip(label_array, length_array))
    dataset = TupleDataset(slice_array, zipped)
    return dataset


def train():
    gpu_id = FLAGS.gpu_id
    if gpu_id >= 0:
        import cupy as cp

    print("Loading data...")
    train_data = load_example(FLAGS.training_data, 10)
    test_data = load_example(FLAGS.eval_data, 10)
    num_classes = get_num_classes(FLAGS.training_data)

    train_dataset = get_tuple_dataset(train_data)
    test_dataset = get_tuple_dataset(test_data)

    print("train_length:", len(train_data["labels"]))
    print("test_length:", len(test_data["labels"]))

    max_epoch = FLAGS.max_epoch
    batch_size = FLAGS.batch_size

    print("Max epoch:", max_epoch, "Batch size:", batch_size)

    print("Building network...")
    network = Classifier(Network(num_classes, gpu_id))

    train_iter = PadIterator(train_dataset, batch_size)
    test_iter = PadIterator(test_dataset, batch_size, False, False)

    if gpu_id >= 0:
        network = network.to_gpu()
        print("Using GPU:", gpu_id)
    else:
        print("Using CPU")

    print("Initializing...")
    optimizer = optimizers.Adam()
    optimizer.setup(network)
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = training.Trainer(updater, stop_trigger=(max_epoch, "epoch"))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        network.predictor, filename='model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(test_iter, network, device=gpu_id))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time'],
                                          log_report=extensions.LogReport(trigger=(1, 'epoch'))))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))
    print("Start training...")
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--training_data",
        type=str,
        default="./data/training.json",
        help="Path to training data")
    parser.add_argument(
        "--eval_data",
        type=str,
        default="./data/eval.json",
        help="Path to evaluation data")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size to use for training/evaluation.")
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=50,
        help="Max epoch to use for training/evaluation.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="result",
        help="Path for storing the model checkpoints.")
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help="Whether use GPU to training/evaluation, negative number stands using CPU.")

    FLAGS, _ = parser.parse_known_args()
    train()

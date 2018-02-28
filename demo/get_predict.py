import os

import chainer
import chainer.functions as F
import numpy as np
from chainer.serializers import load_npz
from rdp import rdp

from predictor import Model as QuickdrawPredictor

classes_file = 'training.json'
model_file = 'model.npz'


def _init_classes_file():
    filename = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), classes_file) + ".classes"
    lines = [line.rstrip() for line in open(filename, 'r').readlines()]
    return len(lines), lines


_num_classes, _class_labels = _init_classes_file()

_predictor = QuickdrawPredictor(_num_classes)
_model_file_path = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), model_file)
load_npz(_model_file_path, _predictor)


def _process_inks_step1(inks):
    drawing = inks["drawing"]
    processed_stroke = []

    for c in drawing:
        stroke_reshape = np.array(c).T
        rdp_stroke = rdp(stroke_reshape, epsilon=2)
        processed_stroke.append(rdp_stroke.T)

    return processed_stroke


def _process_inks_step2(inks):
    """Parse an ndjson line and return ink (as np array) and classname."""
    stroke_lengths = [len(stroke[0]) for stroke in inks]
    total_points = sum(stroke_lengths)
    np_ink = np.zeros((total_points, 3), dtype=np.float32)
    current_t = 0
    for stroke in inks:
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
    return np_ink


def get_predict(inks, topN=3):
    inks = _process_inks_step1(inks)
    inks = _process_inks_step2(inks)

    inks = np.expand_dims(inks, axis=0)
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        y = _predictor(inks)

    target_class_index = F.squeeze(y).data
    target_class_index = target_class_index.argsort()[-topN:][::-1]
    target_class = []
    for c in target_class_index.tolist():
        target_class.append(_class_labels[c])
    return target_class

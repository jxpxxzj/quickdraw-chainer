import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain


class Network(chainer.Chain):
    def __init__(self, num_classes, gpu_id):
        super(Network, self).__init__()
        with self.init_scope():
            self.conv1 = L.ConvolutionND(
                ndim=1, in_channels=3, out_channels=48, ksize=5, stride=1, pad=2)
            self.batch2 = L.BatchNormalization(48)
            self.conv2 = L.ConvolutionND(
                ndim=1, in_channels=48, out_channels=64, ksize=5, stride=1, pad=2)
            self.batch3 = L.BatchNormalization(64)
            self.conv3 = L.ConvolutionND(
                ndim=1, in_channels=64, out_channels=96, ksize=3, stride=1, pad=1)
            self.lstm = L.NStepBiLSTM(
                n_layers=3, in_size=96, out_size=128, dropout=0.3)
            self.fc = L.Linear(in_size=256, out_size=num_classes)
            self.gpu_id = gpu_id

    def __call__(self, x, t):
        h = F.swapaxes(x, 1, 2)
        h = self.conv1(h)
        h = F.dropout(h, 0.3)
        h = self.batch2(h)
        h = self.conv2(h)
        h = F.dropout(h, 0.3)
        h = self.conv3(h)
        h = F.swapaxes(h, 1, 2)
        h = F.separate(h, axis=0)

        _, _, h = self.lstm(None, None, h)

        length_array = F.separate(t, axis=1)[1]

        if self.gpu_id >= 0:
            sequence = [F.squeeze(cp.ones((1, c.data.get()), dtype="bool"))
                        for c in length_array]
        else:
            sequence = [F.squeeze(np.ones((1, c.data), dtype="bool"))
                        for c in length_array]

        sequence_pad = F.pad_sequence(sequence, padding=False)
        expand_dims = F.expand_dims(sequence_pad, 2)
        mask = F.tile(expand_dims, (1, 1, h[0].shape[1]))
        h = F.stack(h)

        if self.gpu_id >= 0:
            arr = cp.zeros(h.shape, dtype="float32")
        else:
            arr = np.zeros(h.shape, dtype="float32")

        zero_outside = F.where(mask, h, arr)
        outputs = F.sum(zero_outside, 1)

        result = self.fc(outputs)
        return result


class Classifier(chainer.Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        x = self.predictor(x, t)
        label = np.transpose(t)[0]
        cross_entropy = F.mean(F.softmax_cross_entropy(x, label))
        accuracy = F.accuracy(x, label)
        chainer.report({'loss': cross_entropy, 'accuracy': accuracy}, self)
        return cross_entropy

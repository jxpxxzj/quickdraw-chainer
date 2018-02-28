import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain


class Model(Chain):
    def __init__(self, num_classes):
        super(Model, self).__init__()
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

    def __call__(self, x):
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
        outputs = F.sum(F.expand_dims(h[0], axis=0), 1)
        result = self.fc(outputs)
        return result

import numpy as np
from chainer import Function as F
from chainer.iterators import SerialIterator


class PadIterator(SerialIterator):
    def next(self):
        batch = self.__next__()
        inks = [a[0] for a in batch]
        label = [a[1] for a in batch]
        inks = F.pad_sequence(inks)
        new_batch = list(zip(inks.data, label))
        return new_batch

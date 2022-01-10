from typing import Union, Tuple

import jittor as jt

from tests.core.dataset import DataSet


class JittorDataSet(DataSet):
    def __init__(self, predict: jt.Var, target: jt.Var):
        """

        :param target: shape: [num_batches, batch_size, ...]
        :param target: shape: [num_batches, batch_size, ...]
        """
        super().__init__(predict, target)

    def __getitem__(self, index: Union[int, slice]) -> Tuple[jt.Var, jt.Var]:
        """
        dataset 里面每个字段，都是 shape: [num_batches, batch_size, ...]
        取某个或者某几个 batch 的数据，返回的字段每个都是 shape: [batch_size*n, ...]
        :param index:
        :return:
        """
        predict = None
        target = None
        if isinstance(index, slice):  # self[start: end: step] 或 self[start: end] 或 self[: end: step] 等
            predict = self.predict[index.start: index.stop: index.step].reshape(-1, *self.predict.shape[2:])
            target = self.target[index.start: index.stop: index.step].reshape(-1, *self.target.shape[2:])
        elif isinstance(index, int):  # self[start]
            predict = self.predict[index]
            target = self.target[index]

        return predict, target

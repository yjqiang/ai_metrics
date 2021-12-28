from typing import Union, Tuple, Any


class DataSet:
    """
    设计原因：测试这里，数据的取法比较单一。
        就是一次取一个 batch 的数据，丢到 execute_evaluate 里；
        或者取好几个 batch 的数据，合成到一起，作为 sklearn 等的输入并结算得到结果后，与本项目的 sync 后的 execute_getmetric 结果进行对比
    """
    def __init__(self, predict: Any, target: Any):
        """

        :param predict: shape: [num_batches, batch_size, ...]
        :param target: shape: [num_batches, batch_size, ...]
        """
        self.predict = predict
        self.target = target

    def __getitem__(self, index: Union[int, slice]) -> Tuple[Any, Any]:
        """
        dataset 里面每个字段，都是 shape: [num_batches, batch_size, ...]
        取某个或者某几个 batch 的数据，返回的字段每个都是 shape: [batch_size*n, ...]
        :param index:
        :return:
        """
        pass

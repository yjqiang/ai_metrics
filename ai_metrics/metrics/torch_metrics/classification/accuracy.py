import torch.utils.data
import torch

from ai_metrics.metric import Metric
from ai_metrics.synchronizers.torch_synchronizer.synchronizer import TorchSynchronizer


class Accuracy(Metric):
    def __init__(self, auto_getmetric_after_evaluate: bool = True, sync_after_evaluate: bool = False):
        super().__init__(
            synchronizer=TorchSynchronizer(),
            auto_getmetric_after_evaluate=auto_getmetric_after_evaluate,
            sync_after_evaluate=sync_after_evaluate
        )

        self.add_element("correct", value=torch.tensor(0), str_aggregate_function="sum")
        self.add_element("total", value=torch.tensor(0), str_aggregate_function="sum")

    def evaluate(self, predict: torch.Tensor, target: torch.Tensor):
        """

        :param predict: shape: [n, ]；每个值表示对当前数据，分类器的预测的分类结果（int 类型, 即 class = 0, 1, ...）
        :param target: shape: [n, ]；每个值表示对当前数据，真值的分类结果（int 类型，即 class = 0, 1, ...）
        :return:
        """
        assert predict.shape == target.shape

        self.elements['correct'].value += torch.sum(torch.eq(predict, target))
        self.elements['total'].value += target.numel()

    def get_metric(self):
        return self.elements['correct'].value.to(torch.float64) / self.elements['total'].value

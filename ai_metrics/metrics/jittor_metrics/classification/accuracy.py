import jittor as jt
import numpy as np

from ai_metrics.metric import Metric
from ai_metrics.synchronizers.jittor_synchronizer.synchronizer import JittorSynchronizer


class Accuracy(Metric):
    def __init__(self, auto_getmetric_after_evaluate: bool = True, sync_after_evaluate: bool = False):
        super().__init__(
            synchronizer=JittorSynchronizer(),
            auto_getmetric_after_evaluate=auto_getmetric_after_evaluate,
            sync_after_evaluate=sync_after_evaluate
        )
        self.add_element("correct", value=jt.array(0).float64(), str_aggregate_function="sum")  # shape: [1,]
        self.add_element("total", value=jt.array(0).float64(), str_aggregate_function="sum")  # shape: [1,]

    def evaluate(self, predict: jt.Var, target: jt.Var) -> None:
        """

        :param predict: shape: [n, ]
        :param target: shape: [n, ]
        :return:
        """

        assert predict.shape == target.shape

        correct: int = np.sum(target.data == predict.data).item()
        total: int = target.numpy().shape[0]
        self.elements['correct'].value += correct
        self.elements['total'].value += total

    def get_metric(self) -> jt.Var:
        return self.elements['correct'].value / self.elements['total'].value

import pytest
import torch
from sklearn.metrics import accuracy_score as sklearn_accuracy

from ai_metrics.metrics.torch_metrics import Accuracy
from tests.torch_tests.core.test_manager import TestManager
from tests.torch_tests.core import utils
from tests.torch_tests.core.dataset import TensorDataSet
from tests.torch_tests.classification import evaluate_params


def _sklearn_accuracy(predict: torch.Tensor, target: torch.Tensor) -> float:
    """

    :param predict: 可以不限设备等；shape: [n, ]；每个值表示对当前数据，分类器的预测的分类结果（int 类型, 即 class = 0, 1, ...）
    :param target: 可以不限设备等shape: [n, ]；shape: [n, ]；每个值表示对当前数据，真值的分类结果（int 类型，即 class = 0, 1, ...）
    :return:
    """
    sklearn_predict = utils.tensor2numpy(predict)
    sklearn_target = utils.tensor2numpy(target)

    return sklearn_accuracy(y_true=sklearn_target, y_pred=sklearn_predict)


@pytest.mark.parametrize(
    "dataset",
    [
        evaluate_params.params_multiclass,
        evaluate_params.params_multiclass,
    ]*2
)
@pytest.mark.parametrize('is_ddp', [True, False])
class TestCases(TestManager):
    @pytest.mark.parametrize('auto_getmetric_after_evaluate', [True, False])
    @pytest.mark.parametrize('sync_after_evaluate', [True, False])
    @pytest.mark.parametrize('need_explicit_to', [True, False])
    def test_accuracy_torch(self, is_ddp: bool, dataset: TensorDataSet, auto_getmetric_after_evaluate: bool, sync_after_evaluate: bool, need_explicit_to: bool) -> None:
        self._test(
            is_ddp=is_ddp,
            need_explicit_to=need_explicit_to,
            dataset=dataset,
            metric_class=Accuracy,
            metric_kwargs={'auto_getmetric_after_evaluate': auto_getmetric_after_evaluate, 'sync_after_evaluate': sync_after_evaluate, 'need_explicit_to': need_explicit_to},
            sklearn_metric=_sklearn_accuracy)

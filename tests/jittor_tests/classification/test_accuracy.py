import pytest
import jittor as jt
from sklearn.metrics import accuracy_score as sklearn_accuracy

from ai_metrics.metrics.jittor_metrics import Accuracy
from tests.jittor_tests.core.test_manager import TestManager
from tests.jittor_tests.core import utils as jittor_utils
from tests.core import utils as utils
from tests.jittor_tests.core.dataset import JittorDataSet
from tests.jittor_tests.classification import evaluate_params


def _sklearn_accuracy(predict: jt.Var, target: jt.Var) -> float:
    """

    :param predict: 可以不限设备等；shape: [n, ]；每个值表示对当前数据，分类器的预测的分类结果（int 类型, 即 class = 0, 1, ...）
    :param target: 可以不限设备等shape: [n, ]；shape: [n, ]；每个值表示对当前数据，真值的分类结果（int 类型，即 class = 0, 1, ...）
    :return:
    """
    sklearn_predict = jittor_utils.jt2numpy(predict)
    sklearn_target = jittor_utils.jt2numpy(target)

    return sklearn_accuracy(y_true=sklearn_target, y_pred=sklearn_predict)


@pytest.mark.parametrize(
    "dataset",
    [
        evaluate_params.params_multiclass,
        evaluate_params.params_multiclass,
    ]*4
)
@pytest.mark.parametrize('is_mpi', [True, False])
class TestCases(TestManager):
    @pytest.mark.parametrize('auto_getmetric_after_evaluate', [True, False])
    @pytest.mark.parametrize('sync_after_evaluate', [True, False])
    def test_accuracy_jittor(self, is_mpi: bool, dataset: JittorDataSet, auto_getmetric_after_evaluate: bool, sync_after_evaluate: bool) -> None:
        self._test(
            is_mpi=is_mpi,
            dataset=dataset,
            metric=Accuracy(auto_getmetric_after_evaluate=auto_getmetric_after_evaluate, sync_after_evaluate=sync_after_evaluate),
            sklearn_metric=_sklearn_accuracy)


if __name__ == '__main__':
    # mpirun 时测试 is_mpi=True 的情况（`mpirun -np 2 python3.7 -m tests.jittor_tests.classification.test_accuracy`），
    # 单设备时测试另一种（`python3.7 -m tests.jittor_tests.classification.test_accuracy`）
    utils.pytest_method(TestCases().test_accuracy_jittor)

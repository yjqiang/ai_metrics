from typing import Callable, Union

import numpy as np
import jittor as jt
import pytest

from ai_metrics import Metric
from tests.jittor_tests.core import utils
from tests.core.dataset import DataSet
from tests.core.utils import is_allclose


jittor_mpi_core = jt.mpi


def _assert_allclose(my_result: jt.Var, sklearn_result: Union[float, np.ndarray], atol: float = 1e-8) -> None:
    """
    测试对比结果，这里不用非得是必须数组且维度对应，一些其他情况例如 np.allclose(np.array([[1e10, ], ]), 1e10+1) 也是 True
    :param my_result: 可以不限设备等
    :param sklearn_result:
    :param atol:
    :return:
    """
    assert is_allclose(utils.jt2numpy(my_result), sklearn_result, atol=atol)


def _test(
        local_rank: int,
        world_size: int,
        dataset: DataSet,
        metric: Metric,
        sklearn_metric: Callable,
        atol: float = 1e-8,
) -> None:
    """

    :param local_rank:
    :param world_size:
    :param dataset: predict 的 shape: [num_batches, batch_size, ...]；target 的 shape: [num_batches, batch_size, ...]
    :param metric:
    :param sklearn_metric: 用于对比 assert 判定的 sklearn 函数
    :param atol: https://pytorch.org/docs/stable/generated/torch.allclose.html#torch.allclose 简单来说就是比较两个 Tensor 是否相等时，认为两两元素的差的绝对值小于一个较小的量值（atol+rtol×∣other∣），即可认为相等
    :return:
    """
    assert len(dataset.predict) == len(dataset.target)
    num_batches = len(dataset.predict)

    # 把数据拆到每个 GPU 上，有点模仿 DistributedSampler 的感觉，但这里数据单位是一个 batch（即每个 i 取了一个 batch 到自己的 GPU 上）
    for i in range(local_rank, num_batches, world_size):
        my_result = metric.execute_evaluate(*dataset[i])

        # TODO: 这个删除就挂了离谱（MPI 模式下），这应该时 jittor 的 bug
        # 如果删了 str(my_result)，则 rank != 0 的 i 不会等 rank=0 的设备
        # print(f'local_rank={local_rank}(world_size={world_size}) is testing i={i}')
        if my_result is not None and local_rank != 0 and True:
            str(my_result)

        if metric.auto_getmetric_after_evaluate:  # 如果设置了 auto_getmetric_after_evaluate 为 True，则 my_result 是有返回值的，因为自动执行了 execute_get_metric
            if metric.sync_after_evaluate and local_rank == 0:
                # dataset[i: i+world_size] 是本轮次（每个轮次，各个 GPU 取到自己的那个 batch 大小的数据）中，所有 GPU 的刚刚取到的数据合集
                # dataset[:i] 是在本轮次之前，所有 GPU 的已经取到的数据合集
                using_predict, using_target = dataset[: i+world_size]  # shape: [n * batch_size, ...]

                sklearn_result = sklearn_metric(using_predict, using_target)
                _assert_allclose(my_result, sklearn_result, atol=atol)

            elif not metric.sync_after_evaluate:  # 自动执行 execute_get_metric 时候，各设备不同步
                # 截止到目前为止（包括本轮次），本 GPU 拥有 [local_rank, local_rank+world_size*1, local_rank+world_size*2, ..., i] 这些 indices 对应的数据
                using_predict, using_target = dataset[local_rank: i + world_size: world_size]  # shape: [n * batch_size, ...]

                sklearn_result = sklearn_metric(using_predict, using_target)
                _assert_allclose(my_result, sklearn_result, atol=atol)

    my_result = metric.execute_get_metric()
    using_predict, using_target = dataset[:]  # 借 __getitem__ 方法取处理数据；shape: [num_batches*batch_size, ...]

    sklearn_result = sklearn_metric(using_predict, using_target)
    _assert_allclose(my_result, sklearn_result, atol=atol)


class TestManager:
    @staticmethod
    def _test(is_mpi: bool, dataset: DataSet, metric: Metric, sklearn_metric: Callable) -> None:
        # mpirun 时测试 is_mpi=True 的情况（`mpirun -np 2 python3.7 -m tests.jittor_tests.classification.test_accuracy`），
        # 单设备时测试另一种（`python3.7 -m tests.jittor_tests.classification.test_accuracy`）
        # 否则 is_mpi 为 False 时，metric 探测到 is_distributed 为 True，自动同步了
        # 反正逻辑比较乱
        if is_mpi == utils.is_distributed():
            _test(
                local_rank=jittor_mpi_core.local_rank(),
                world_size=jittor_mpi_core.world_size(),
                dataset=dataset,
                metric=metric,
                sklearn_metric=sklearn_metric
            )
            return
        pytest.skip('is_mpi != utils.is_distributed()')

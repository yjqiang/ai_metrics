import sys
import os
from functools import partial
from typing import Type, Callable, Union, Dict, Any
import copy

import pytest
import numpy as np
import torch
import torch.distributed
from torch.multiprocessing import Pool, set_start_method

from ai_metrics import Metric
from tests.torch_tests.core import utils
from tests.core.dataset import DataSet
from tests.core.utils import is_allclose


try:
    set_start_method("spawn")
except RuntimeError:
    pass

NUM_PROCESSES = 2


def _assert_allclose(my_result: torch.Tensor, sklearn_result: Union[float, np.ndarray], atol: float = 1e-8) -> None:
    """
    测试对比结果，这里不用非得是必须数组且维度对应，一些其他情况例如 np.allclose(np.array([[1e10, ], ]), 1e10+1) 也是 True
    :param my_result: 可以不限设备等
    :param sklearn_result:
    :param atol:
    :return:
    """
    assert is_allclose(utils.tensor2numpy(my_result), sklearn_result, atol=atol)


def _test(
        local_rank: int,
        world_size: int,
        device: torch.device,
        need_explicit_to: bool,
        dataset: DataSet,
        metric_class: Type[Metric],
        metric_kwargs: Dict[str, Any],
        sklearn_metric: Callable,
        atol: float = 1e-8,
) -> None:
    """

    :param local_rank:
    :param world_size:
    :param device:
    :param need_explicit_to: 在 torch 中，是否需要显式地执行 metric.to(device) 操作
    :param dataset: predict 的 shape: [num_batches, batch_size, ...]；target 的 shape: [num_batches, batch_size, ...]
    :param metric_class:
    :param metric_kwargs:
    :param sklearn_metric: 用于对比 assert 判定的 sklearn 函数
    :param atol: https://pytorch.org/docs/stable/generated/torch.allclose.html#torch.allclose 简单来说就是比较两个 Tensor 是否相等时，认为两两元素的差的绝对值小于一个较小的量值（atol+rtol×∣other∣），即可认为相等
    :return:
    """

    # metric 应该是每个进程有自己的一个 instance，所以在 _test 里面实例化
    metric = metric_class(**metric_kwargs)
    # dataset 也类似（每个进程有自己的一个）
    dataset = copy.deepcopy(dataset)

    # move to device
    if need_explicit_to:
        metric.to(device)
    dataset.to(device)

    assert len(dataset.predict) == len(dataset.target)
    num_batches = len(dataset.predict)

    # 把数据拆到每个 GPU 上，有点模仿 DistributedSampler 的感觉，但这里数据单位是一个 batch（即每个 i 取了一个 batch 到自己的 GPU 上）
    for i in range(local_rank, num_batches, world_size):
        my_result = metric.execute_evaluate(*dataset[i])

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


def setup_ddp(rank: int, world_size: int, master_port: int) -> None:
    """Setup ddp environment."""

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


class TestManager:
    """
    设计原因：pytest 应该就没有为这种需求设计，即多个 test 共享同一个全局数据（ddp 初始化花费大量时间）；而这里为了实现这个需求，我们不得不借助 setup_class 去生成共享的东西（作为 class 变量）
        我觉得这个设计比较丑，但是为了性能，不得不这样
    """
    pool: Pool

    @staticmethod
    def setup_class(cls: Type['TestManager']) -> None:
        processes = NUM_PROCESSES
        cls.pool = Pool(processes=processes)
        master_port = utils.find_free_network_port()
        cls.pool.starmap(setup_ddp, [(rank, processes, master_port) for rank in range(processes)])

    @staticmethod
    def teardown_class(cls: Type['TestManager']) -> None:
        cls.pool.close()
        cls.pool.join()

    def _test(self, is_ddp: bool, need_explicit_to: bool, dataset: DataSet, metric_class: Type[Metric], metric_kwargs: Dict[str, Any], sklearn_metric: Callable) -> None:
        if is_ddp:
            if sys.platform == "win32":
                pytest.skip("DDP not supported on windows")

            processes = NUM_PROCESSES
            self.pool.starmap(
                partial(
                    _test,
                    need_explicit_to=need_explicit_to,
                    dataset=dataset,
                    metric_class=metric_class,
                    metric_kwargs=metric_kwargs,
                    sklearn_metric=sklearn_metric,
                ),
                [(rank, processes, torch.device(f'cuda:{rank}')) for rank in range(processes)]
            )
        else:
            device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
            _test(
                local_rank=0,
                world_size=1,
                device=device,
                need_explicit_to=need_explicit_to,
                dataset=dataset,
                metric_class=metric_class,
                metric_kwargs=metric_kwargs,
                sklearn_metric=sklearn_metric
            )

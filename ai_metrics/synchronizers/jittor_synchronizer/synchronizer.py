import copy
from typing import Any, Callable

import jittor as jt

from ai_metrics.synchronizers.synchronizer import Synchronizer, Element


def default_deepcopy(data: Any) -> Any:
    return copy.deepcopy(data)


class JittorElement(Element):
    """
    设计原因：一个 metric 的计算元素，这里仅仅是为了方便存储一些东西（例如 'sum' -> dim_zero_sum）以及指明这是属于 torch synchronizer 控制的
    而 value 不一定是 Tensor 的

    """
    def __init__(self, name: str, value: Any, aggregate_function: Callable, deepcopy_function: Callable[[Any], Any] = default_deepcopy):
        """

        :param name:
        :param value: 初始化值：一般为 0
        :param aggregate_function: torch 情景下，把某元素从所有的 device 进行 gather 操作后，会成为 list[torch.Tensor]，通过 aggregate_function 去把这些数据转为 torch.Tensor（和单设备一致）
        """
        super().__init__(name, value, deepcopy_function)

        self.aggregate_function = aggregate_function

    def to(self, target: Any) -> None:
        """
        啥也不做（为了使得 api 与 torch 一致）
        :param target:
        :return:
        """
        pass

    def auto_to(self, target: Any) -> None:
        """
        啥也不做（为了使得 api 一致）
        :param target:
        :return:
        """
        pass


class JittorSynchronizer(Synchronizer):
    @staticmethod
    def create_element(name: str, value: Any, str_aggregate_function: str) -> JittorElement:
        """
        对标 torchmetrics/metric.py add_state 函数
        :param name:
        :param value:
        :param str_aggregate_function:
        :return:
        """
        if str_aggregate_function == "sum":
            aggregate_function = JittorSynchronizer.dim_zero_sum
        else:
            aggregate_function = None
        element = JittorElement(name, value, aggregate_function)
        return element

    @staticmethod
    def is_distributed() -> bool:
        """
        同 torchmetrics 中 jit_distributed_available
        :return:
        """
        return jt.in_mpi

    @staticmethod
    def sync(element: JittorElement) -> None:
        if isinstance(element.value, jt.Var):
            value = element.value
            element.value = element.aggregate_function(value) if element.aggregate_function is not None else value

    @staticmethod
    def dim_zero_sum(x: jt.Var) -> jt.Var:
        return x.mpi_all_reduce('add')

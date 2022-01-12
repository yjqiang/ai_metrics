from typing import List, Optional, Any, Callable

import torch
import torch.distributed
import torch.nn.functional as F

from ai_metrics.synchronizers.synchronizer import Synchronizer, Element
from ai_metrics import utils


def _simple_gather_all_tensors(result: torch.Tensor, group: Any, world_size: int) -> List[torch.Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result, group)
    return gathered_result


def default_to_function(data: Any, device: torch.device) -> Any:
    """
    默认利用 apply_to_collection 对 value 中所有为 torch.Tensor 的元素全部执行 x.to(device) 操作

    :param data:
    :param device:
    :return:
    """
    return utils.apply_to_collection(data, torch.Tensor, lambda x: x.to(device))


def default_auto_to_function(data: Any, target: Any) -> Any:
    """
    默认利用 apply_to_collection 对 value 中所有为 torch.Tensor 的元素全部执行 x.to(real_target) 操作

    :param data:
    :param target:
    :return:
    """
    # target 可能是 List[torch.Tensor]，这时候我们认为里面所有的 torch.Tensor 都是类同的
    real_target = utils.check_collection(target, torch.Tensor, lambda x: x)
    # https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to
    # 如果 real_target is None，此时 If dtype is None it is inferred to be self.dtype. 不影响什么结果
    return utils.apply_to_collection(data, torch.Tensor, lambda x: x.to(real_target))


class TorchElement(Element):
    """
    设计原因：一个 metric 的计算元素，这里仅仅是为了方便存储一些东西（例如 'sum' -> dim_zero_sum）以及指明这是属于 torch synchronizer 控制的
    而 value 不一定是 Tensor 的

    """
    def __init__(self, name: str, value: Any, aggregate_function: Callable,
                 to_function: Callable[[Any, torch.device], Any] = default_to_function, auto_to_function: Callable[[Any, Any], Any] = default_auto_to_function) -> None:
        """

        :param name: 见基类
        :param value: 见基类
        :param aggregate_function: torch 情景下，把某元素从所有的 device 进行 gather 操作后，会成为 list[torch.Tensor]，通过 aggregate_function 去把这些数据转为 torch.Tensor（和单设备一致）
        :param to_function: 对应 Metric(...).to 函数，负责具体执行 to_function(data, device)
        :param auto_to_function: 对应 Metric(...).auto_to 函数，负责具体执行 auto_to_function(data, target)
        """
        super().__init__(name, value)

        self.aggregate_function = aggregate_function
        self.to_function = to_function
        self.auto_to_function = auto_to_function

    def to(self, device: torch.device) -> None:
        """
        详见初始化时候的相关参数介绍

        :param device:
        :return:
        """
        self.value = self.to_function(self.value, device)

    def auto_to(self, target: Any) -> None:
        """
        详见初始化时候的相关参数介绍

        :param target:
        :return:
        """
        self.value = self.auto_to_function(self.value, target)


class TorchSynchronizer(Synchronizer):
    @staticmethod
    def dim_zero_sum(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=0)

    @staticmethod
    def _gather_all(result: torch.Tensor, group: Optional[Any] = None) -> List[torch.Tensor]:
        """Function to gather all tensors from several ddp processes onto a list that is broadcasted to all processes.
        Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
        tensors are padded, gathered and then trimmed to secure equal workload for all processes.

        Args:
            result: the value to sync
            group: the process group to gather results from. Defaults to all processes (world)

        Return:
            gathered_result: list with size equal to the process group where
                gathered_result[i] corresponds to result tensor from process i
        """
        if group is None:
            group = torch.distributed.group.WORLD

        # convert tensors to contiguous format
        result = result.contiguous()

        world_size = torch.distributed.get_world_size(group)
        torch.distributed.barrier(group=group)

        # if the tensor is scalar, things are easy
        if result.ndim == 0:
            return _simple_gather_all_tensors(result, group, world_size)

        # 1. Gather sizes of all tensors
        local_size = torch.tensor(result.shape, device=result.device)
        local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(local_sizes, local_size, group=group)
        max_size = torch.stack(local_sizes).max(dim=0).values
        all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

        # 2. If shapes are all the same, then do a simple gather:
        if all_sizes_equal:
            return _simple_gather_all_tensors(result, group, world_size)

        # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
        pad_dims = []
        pad_by = (max_size - local_size).detach().cpu()
        for val in reversed(pad_by):
            pad_dims.append(0)
            pad_dims.append(val.item())
        result_padded = F.pad(result, pad_dims)
        gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_result, result_padded, group)
        for idx, item_size in enumerate(local_sizes):
            slice_param = [slice(dim_size) for dim_size in item_size]
            gathered_result[idx] = gathered_result[idx][slice_param]
        return gathered_result

    @staticmethod
    def to(element: TorchElement, device: torch.device) -> None:
        """
        对应 Metric(...).to 函数

        :param element:
        :param device:
        :return:
        """
        element.to(device)

    @staticmethod
    def auto_to(element: TorchElement, target: Any) -> None:
        """
        对应 Metric(...).auto_to 函数

        :param element:
        :param target:
        :return:
        """
        element.auto_to(target)

    @staticmethod
    def sync(element: TorchElement) -> None:
        if isinstance(element.value, torch.Tensor):
            value = TorchSynchronizer._gather_all(element.value)
            if isinstance(value[0], torch.Tensor):
                value = torch.stack(value)
            element.value = element.aggregate_function(value) if element.aggregate_function is not None else value

    @staticmethod
    def create_element(name: str, value: Any, str_aggregate_function: str) -> TorchElement:
        """
        对标 torchmetrics/metric.py add_state 函数
        :param name:
        :param value:
        :param str_aggregate_function:
        :return:
        """
        if str_aggregate_function == "sum":
            aggregate_function = TorchSynchronizer.dim_zero_sum
        else:
            aggregate_function = None
        element = TorchElement(name, value, aggregate_function)
        return element

    @staticmethod
    def is_distributed() -> bool:
        """
        同 torchmetrics 中 jit_distributed_available
        :return:
        """
        return torch.distributed.is_available() and torch.distributed.is_initialized()

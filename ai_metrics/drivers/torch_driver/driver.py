from typing import List, Optional, Any, Callable

import torch
import torch.distributed
import torch.nn.functional as F

from ai_metrics.drivers.base_driver import BaseDriver, BaseElement


def _simple_gather_all_tensors(result: torch.Tensor, group: Any, world_size: int) -> List[torch.Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result, group)
    return gathered_result


class TorchElement(BaseElement):
    """
    设计原因：一个 metric 的计算元素，这里仅仅是为了方便存储一些东西（例如 'sum' -> dim_zero_sum）以及指明这是属于 torch driver 控制的
    而 value 不一定是 Tensor 的

    """
    def __init__(self, name: str, value: Any, aggregate_function: Callable):
        """

        :param name:
        :param value: 初始化值：一般为 0
        :param aggregate_function: torch 情景下，把某元素从所有的 device 进行 gather 操作后，会成为 list[torch.Tensor]，通过 aggregate_function 去把这些数据转为 torch.Tensor（和单设备一致）
        """
        super().__init__(name, value)

        self.aggregate_function = aggregate_function


class TorchDriver(BaseDriver):
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
    def to_device(element: TorchElement, device: torch.device) -> None:
        if isinstance(element.value, torch.Tensor):
            element.value = element.value.to(device)

    @staticmethod
    def sync(element: TorchElement) -> None:
        if isinstance(element.value, torch.Tensor):
            value = TorchDriver._gather_all(element.value)
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
            aggregate_function = TorchDriver.dim_zero_sum
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

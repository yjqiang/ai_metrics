from typing import Any

import torch

from ai_metrics.metric import Metric
from ai_metrics.synchronizers.torch_synchronizer.synchronizer import TorchSynchronizer


TARGET_DEVICE = torch.device("cuda", 0)
ORIGINAL_DEVICE = torch.device("cpu")


def test_reset():
    """
    测试 reset：包括 init_value 是否在本过程保存不变，以及 device 迁移等问题
    :return:
    """
    class A(Metric):
        def __init__(self):
            super().__init__(
                synchronizer=TorchSynchronizer(),
                need_explicit_to=True
            )
            self.add_element("x", [1, 2, torch.tensor([2.0], device=ORIGINAL_DEVICE)], 'sum')

        def evaluate(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get_metric(self) -> Any:
            pass

    a = A()
    a.to(device=TARGET_DEVICE)
    assert a.elements['x'].value == [1, 2, torch.tensor([2.0], device=TARGET_DEVICE)]
    assert a.elements['x'].init_value == [1, 2, torch.tensor([2.0], device=ORIGINAL_DEVICE)]

    # https://discuss.pytorch.org/t/what-is-in-place-operation/16244/3
    a.elements['x'].value[2].add_(torch.tensor([1.0], device=TARGET_DEVICE))
    a.elements['x'].value.append(4)
    assert a.elements['x'].value == [1, 2, torch.tensor([3.0], device=TARGET_DEVICE), 4]

    a.reset()
    assert a.elements['x'].value == [1, 2, torch.tensor([2.0], device=TARGET_DEVICE)]
    assert a.elements['x'].init_value == [1, 2, torch.tensor([2.0], device=ORIGINAL_DEVICE)]

from typing import Any
from abc import ABC, abstractmethod


class Element:
    """
    设计原因：一个 metric 的计算元素

    """
    def __init__(self, name: str, value: Any):
        """

        :param name:
        :param value: 初始化值：一般为 0
        """
        self.name = name
        self.value = value


class Synchronizer(ABC):
    """
    详细例子均请见 synchronizers/torch/synchronizer.py
    """
    @staticmethod
    @abstractmethod
    def is_distributed() -> bool:
        pass

    @staticmethod
    @abstractmethod
    def sync(x: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def to_device(element: Element, device: Any) -> None:
        pass

    @staticmethod
    def create_element(name: str, value: Any, str_aggregate_function: str) -> Element:
        pass

    @staticmethod
    @abstractmethod
    def dim_zero_sum(x: Any) -> Any:
        """

        :param x:
        :return:
        """
        pass

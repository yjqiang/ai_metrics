from typing import Any, List
from abc import ABC, abstractmethod


class BaseElement:
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


class BaseDriver(ABC):
    """
    详细例子均请见 drivers/torch/driver.py
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
    def to_device(element: BaseElement, device: Any) -> None:
        pass

    @staticmethod
    def create_element(name: str, value: Any, str_aggregate_function: str) -> BaseElement:
        pass

    @staticmethod
    @abstractmethod
    def dim_zero_sum(x: Any) -> Any:
        """

        :param x:
        :return:
        """
        pass

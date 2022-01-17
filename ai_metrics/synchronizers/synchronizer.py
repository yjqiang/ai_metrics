from typing import Any, Callable
from abc import ABC, abstractmethod


class Element:
    """
    设计原因：一个 metric 的计算元素

    """
    def __init__(self, name: str, value: Any, deepcopy_function: Callable[[Any], Any]):
        """

        :param name:
        :param value: 初始化值：一般为 0
        """
        self.name = name
        self.value = value
        self.init_value = deepcopy_function(value)  # 保存初始化的值，只读的
        self.context = None  # 命名参考：进程“上下文”、中断“上下文”
        self.deepcopy_function = deepcopy_function
        self.target = None  # 给 self.to 用

    def deepcopy(self, data: Any) -> Any:
        """
        设计原因：用于 reset 等时候，相互赋值
        :param data:
        :return:
        """
        return self.deepcopy_function(data)

    def reset(self) -> None:
        self.value = self.deepcopy(self.init_value)

    def save_before_sync(self) -> None:
        """
        用于 sync 和 unsync
        :return:
        """
        self.context = self.value

    def reload_in_unsync(self) -> None:
        """
        用于 sync 和 unsync
        :return:
        """
        self.value = self.context

    @abstractmethod
    def to(self, target: Any) -> None:
        pass

    @abstractmethod
    def auto_to(self, target: Any) -> None:
        pass


class Synchronizer(ABC):
    """
    详细例子均请见 synchronizers/torch/synchronizer.py
    """
    @staticmethod
    def create_element(name: str, value: Any, str_aggregate_function: str) -> Element:
        pass

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
    def dim_zero_sum(x: Any) -> Any:
        """

        :param x:
        :return:
        """
        pass

from contextlib import contextmanager
from typing import Optional, Any, Dict, Generator

from ai_metrics.synchronizers.synchronizer import Synchronizer, Element


class Metric:
    def __init__(self, synchronizer: Optional[Synchronizer] = None):
        """

        :param synchronizer: 用户可以自定义 synchronizer，如果为 None，则会自动 detect
        """
        # TODO: auto detect synchronizer
        self.synchronizer = synchronizer

        self.elements: Dict[str, Element] = {}

        self.need_get_metric_after_every_evaluate = True
        self.need_sync = True  # 需要在各个设备之间进行同步（即使处于 ddp 状态下，也可以选择各个 gpu 之间不同步）

    def add_specific_element(self, name: str, element: Element) -> None:
        self.elements[name] = element

    def add_element(self, name: str, value: Any, str_aggregate_function: str) -> None:
        """
        推荐的添加方法，可以自动检测一些东西
        :param name:
        :param value:
        :param str_aggregate_function:
        :return:
        """
        element = self.synchronizer.create_element(name, value, str_aggregate_function)
        self.add_specific_element(name, element)

    def to(self, device) -> None:
        for element in self.elements.values():
            self.synchronizer.to_device(element, device)

    def sync(self, need_sync: bool) -> None:
        if not need_sync or not self.synchronizer.is_distributed():
            return

        # 执行把数据从多设备进行同步过程
        for element in self.elements.values():
            self.synchronizer.sync(element)

    def unsync(self, need_unsync: bool) -> None:
        if not need_unsync:
            return

    @contextmanager
    def sync_context(self, need_sync: bool, need_unsync: bool) -> Generator:
        self.sync(
            need_sync=need_sync,
        )

        yield

        self.unsync(need_unsync=need_unsync)

    def evaluate(self, *args: Any, **kwargs: Any) -> None:
        """
        用户自定义 Metric 的函数，迭代
        :return:
        """
        pass

    def execute_evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """
        用户使用 Metric，参数与 evaluate 一致
        :param args:
        :param kwargs:
        :return:
        """
        self.evaluate(*args, **kwargs)

        if self.need_get_metric_after_every_evaluate:
            self.need_sync = False
            value = self.get_metric()
            self.need_sync = True
            return value
        return None

    def get_metric(self) -> Any:
        """
        用户自定义 Metric 的函数，获取最终结果
        :return:
        """
        pass

    def execute_get_metric(self) -> Any:
        """
        用户使用 Metric，参数与 get_metric 一致
        :return:
        """
        with self.sync_context(
                need_sync=self.need_sync,
                need_unsync=self.need_sync
        ):
            value = self.get_metric()
        return value

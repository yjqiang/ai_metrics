from contextlib import contextmanager
from typing import Optional, Any, Dict, Generator

from ai_metrics.drivers.base_driver import BaseDriver, BaseElement


class BaseMetric:
    def __init__(self, driver: Optional[BaseDriver] = None):
        """

        :param driver: 用户可以自定义 driver，如果为 None，则会自动 detect
        """
        # TODO: auto detect driver
        self.driver = driver

        self.elements: Dict[str, BaseElement] = {}

        self.need_compute_after_every_update = True
        self.need_sync = True  # 需要在各个设备之间进行同步（即使处于 ddp 状态下，也可以选择各个 gpu 之间不同步）

    def add_specific_element(self, name: str, element: BaseElement) -> None:
        self.elements[name] = element

    def add_element(self, name: str, value: Any, str_aggregate_function: str) -> None:
        """
        推荐的添加方法，可以自动检测一些东西
        :param name:
        :param value:
        :param str_aggregate_function:
        :return:
        """
        element = self.driver.create_element(name, value, str_aggregate_function)
        self.add_specific_element(name, element)

    def to(self, device) -> None:
        for element in self.elements.values():
            self.driver.to_device(element, device)

    def sync(self, need_sync: bool) -> None:
        if not need_sync or not self.driver.is_distributed():
            return

        # 执行把数据从多设备进行同步过程
        for element in self.elements.values():
            self.driver.sync(element)

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

    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        用户自定义 Metric 的函数，迭代
        :return:
        """
        pass

    def execute_update(self, *args: Any, **kwargs: Any) -> Any:
        """
        用户使用 Metric，参数与 update 一致
        :param args:
        :param kwargs:
        :return:
        """
        self.update(*args, **kwargs)

        if self.need_compute_after_every_update:
            self.need_sync = False
            value = self.compute()
            self.need_sync = True
            return value
        return None

    def compute(self) -> Any:
        """
        用户自定义 Metric 的函数，获取最终结果
        :return:
        """
        pass

    def execute_compute(self) -> Any:
        """
        用户使用 Metric，参数与 compute 一致
        :return:
        """
        with self.sync_context(
                need_sync=self.need_sync,
                need_unsync=self.need_sync
        ):
            value = self.compute()
        return value

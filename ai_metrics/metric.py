from contextlib import contextmanager
from typing import Any, Dict, Generator

from ai_metrics.synchronizers.synchronizer import Synchronizer, Element


class Metric:
    def __init__(self, synchronizer: Synchronizer, auto_getmetric_after_evaluate: bool = True, sync_after_evaluate: bool = False):
        """

        :param synchronizer: 用户可以自定义 synchronizer
        :param auto_getmetric_after_evaluate: 在每次执行 execute_evaluate 后，是否需要返回 metric 的计算结果（execute_evaluate 往往仅仅是对一些数值的累计，例如 correct、total）
            置为 True 后，自动额外执行 execute_get_metric 来得到截止到目前为止的结果（例如 accuracy）
        :param sync_after_evaluate: auto_getmetric_after_evaluate 为 True 后，自动额外执行 execute_get_metric 时，是否需要执行同步
        """
        self.synchronizer = synchronizer

        self.elements: Dict[str, Element] = {}

        self.auto_getmetric_after_evaluate = auto_getmetric_after_evaluate

        # 注意：这两个变量不要在 class 之外随意修改，否则一会儿同步，一会儿不同步，可能数值混乱
        self.sync_after_evaluate = sync_after_evaluate
        self.need_sync = True  # 需要在各个设备之间进行同步（即使处于 ddp 状态下，也可以选择各个 gpu 之间不同步）

        # 表示是否需要执行还原操作；因为 sync 的判定条件比较多，有这个变量可以方便确定 sync 中是否执行了各设备同步操作，若执行了，需要在 unsync 时候进行恢复操作，复原到未同步状态
        self._did_sync_successfully = False

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

        # 保存上下文
        # 在 unsync 时候，需要还原
        # 否则多 GPU 在 sync_after_evaluate 时，多 GPU 当前所有数据合起来算了一个 metric 结果
        # 在之后进行一些操作后，还有同步过程，那么数据会重复合计
        for element in self.elements.values():
            element.save_before_sync()

        # 执行把数据从多设备进行同步过程
        for element in self.elements.values():
            self.synchronizer.sync(element)

        self._did_sync_successfully = True

    def unsync(self, need_unsync: bool) -> None:
        if not need_unsync or not self._did_sync_successfully:
            return

        # 执行把数据进行还原过程
        for element in self.elements.values():
            element.reload_in_unsync()

        self._did_sync_successfully = False

    @contextmanager
    def sync_context(self) -> Generator:
        self.sync(
            need_sync=self.need_sync,
        )

        yield

        self.unsync(need_unsync=self.need_sync)

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

        if self.auto_getmetric_after_evaluate:
            self.need_sync = self.sync_after_evaluate
            value = self.execute_get_metric()
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
        with self.sync_context():
            value = self.get_metric()
        return value

# TODO  delete; evaluate => evaluate; get_metric => get_metric; BLEU; syncrizer


"""
mpirun -np 2 python3.7 -m ai_metrics.demo_jt_mpi
"""
import jittor as jt
import numpy as np

from ai_metrics.metric import Metric
from ai_metrics.synchronizers.jittor_synchronizer.synchronizer import JittorSynchronizer


# 开启 GPU 加速
jt.flags.use_cuda = 1

jittor_mpi_core = jt.mpi


class MyAccuracy(Metric):
    def __init__(self):
        super().__init__(synchronizer=JittorSynchronizer())

        self.add_element("correct", value=jt.array(0).float64(), str_aggregate_function="sum")  # shape: [1,]
        self.add_element("total", value=jt.array(0).float64(), str_aggregate_function="sum")  # shape: [1,]

    def evaluate(self, predict: jt.Var, target: jt.Var):
        """

        :param predict: shape: [n, ]
        :param target: shape: [n, ]
        :return:
        """

        assert predict.shape == target.shape

        correct: int = np.sum(target.data == predict.data).item()
        total: int = target.numpy().shape[0]
        self.elements['correct'].value += correct
        self.elements['total'].value += total

    def get_metric(self):
        return self.elements['correct'].value / self.elements['total'].value


def main():
    accuracy = MyAccuracy()

    # torchmetrics/metric.py Metric 继承了 torch.nn.Module，且 metric 是 model 的一个实例的参数，所以在执行 model.to(device) 会执行 metric._apply
    # 里面有 if isinstance(current_val, Tensor): setattr(this, key, fn(current_val))
    # 把数据转为 device 的数据
    accuracy.to(None)
    if jittor_mpi_core.local_rank() == 0:
        preds = jt.array([1, 1, 0])
        target = jt.array([1, 1, 0])

    else:
        preds = jt.array([3, 1, 3])
        target = jt.array([3, 1, 0])

    acc = accuracy.execute_evaluate(preds, target)
    acc_expected = np.sum(target.data == preds.data).item() / target.numpy().shape[0]
    print(f'RANK: {jittor_mpi_core.local_rank()} acc: {acc} acc_expected: {acc_expected}')

    acc = accuracy.execute_get_metric()
    print(f'RANK: {jittor_mpi_core.local_rank()} acc: {acc} acc_expected: {5 / 6}')


if __name__ == "__main__":
    main()

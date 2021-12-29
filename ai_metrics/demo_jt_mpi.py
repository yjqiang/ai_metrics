"""
mpirun -np 2 python3.7 -m ai_metrics.demo_jt_mpi
"""
import jittor as jt
import numpy as np

from ai_metrics.metrics.jittor_metrics import Accuracy


# 开启 GPU 加速
jt.flags.use_cuda = 1

jittor_mpi_core = jt.mpi


def main():
    accuracy = Accuracy()
    accuracy.to(None)

    # shape: [n=2, batch_size] 表示对于每个设备，每个 epoch 有 n/2 次 metric 使用，每次都是一个 batch_size（一共两个设备）
    predict = jt.array([[1, 1, 0], [3, 1, 3]])
    target = jt.array([[1, 1, 0], [3, 1, 0]])

    using_predict = predict[jittor_mpi_core.local_rank()]
    using_target = target[jittor_mpi_core.local_rank()]

    acc = accuracy.execute_evaluate(using_predict, using_target)
    acc_expected = np.sum(using_target.data == using_predict.data).item() / using_target.data.shape[0]
    print(f'RANK: {jittor_mpi_core.local_rank()} acc: {acc} acc_expected: {acc_expected}')

    acc = accuracy.execute_get_metric()
    acc_expected = np.sum(predict.data == target.data).item() / (target.data.shape[0] * target.data.shape[1])
    print(f'RANK: {jittor_mpi_core.local_rank()} acc: {acc} acc_expected: {acc_expected}')


if __name__ == "__main__":
    main()

import jittor as jt
import numpy as np


def jt2numpy(x: jt.Var) -> np.ndarray:
    """
    转为 numpy
    :param x: 可以不限设备等
    :return:
    """
    return x.numpy()


def is_distributed() -> bool:
    """
    是否在 mpi 状态下
    :return:
    """
    return jt.in_mpi

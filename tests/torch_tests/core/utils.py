import torch
import numpy as np


def tensor2numpy(x: torch.Tensor) -> np.ndarray:
    """
    利用 detach 生成一个与 current graph 无关的 tensor（Returned Tensor shares the same storage with the original one），
    然后 cpu 转化设备，最后转为 numpy
    注意：tensor.detach().cpu() 不一定执行 copy 操作（Tensor.cpu 提及 If this object is already in CPU memory and on the correct device, then no copy is performed and the original object is returned）
    :param x: 可以不限设备等
    :return:
    """
    return x.detach().cpu().numpy()

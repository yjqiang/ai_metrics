from tests.torch_tests.core.dataset import TensorDataSet

import torch


NUM_BATCHES = 36  # 如果有多个 gpu，尽量可以正好分开
BATCH_SIZE = 32
NUM_CLASSES = 5


# 设计原因：不少 metric 的输入一致的，仅仅是不同的评估方法，比如 accuracy 和 f1
params_multiclass = TensorDataSet(
    predict=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
)

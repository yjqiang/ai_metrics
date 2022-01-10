from tests.jittor_tests.core.dataset import JittorDataSet

import jittor as jt


NUM_BATCHES = 10  # 如果有多个 gpu，尽量可以正好分开
BATCH_SIZE = 32
NUM_CLASSES = 5

jt.misc.set_global_seed(seed=986, different_seed_for_mpi=False)  # 不同的 device 同步数据，不然数据不一致，肯定计算错误

# 设计原因：不少 metric 的输入一致的，仅仅是不同的评估方法，比如 accuracy 和 f1
params_multiclass = JittorDataSet(
    predict=jt.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
    target=jt.randint(low=0, high=NUM_CLASSES, shape=(NUM_BATCHES, BATCH_SIZE)),
)

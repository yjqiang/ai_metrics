import os
import subprocess
import sys
import __main__

import torch.distributed as dist
import torch
import torch.utils.data
import torch
from torch import device

from ai_metrics.base_metric import BaseMetric
from ai_metrics.drivers.torch_driver.driver import TorchDriver


class MyAccuracy(BaseMetric):
    def __init__(self, device):
        super().__init__(driver=TorchDriver(), device=device)

        self.add_element("correct", value=torch.tensor(0), str_aggregate_function="sum")
        self.add_element("total", value=torch.tensor(0), str_aggregate_function="sum")

    def update(self, predict: torch.Tensor, target: torch.Tensor):
        """

        :param predict: shape: [n, ]
        :param target: shape: [n, ]
        :return:
        """
        assert predict.shape == target.shape

        self.elements['correct'].value += torch.sum(torch.eq(predict, target))
        self.elements['total'].value += target.numel()

    def compute(self):
        return self.elements['correct'].value.float() / self.elements['total'].value


WORLD_SIZE = 2
torch.set_printoptions(linewidth=200)


class Environment:
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    def creates_processes_externally(self) -> bool:
        """Returns whether the cluster creates the processes or not.
        If at least :code:`LOCAL_RANK` is available as environment variable, Lightning assumes the user acts as the
        process launcher/job scheduler and Lightning will not launch new processes.
        """
        return "LOCAL_RANK" in os.environ


ENV = Environment()

class DDP:
    def __init__(self):
        self.interactive_ddp_procs = []

    def call_children_scripts(self):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        # 找到 是怎么启动的本程序（再用同样的方法启动剩下 WORLD_SIZE-1 个相同进程）
        if __main__.__spec__ is None:  # pragma: no-cover
            # Script called as `python a/b/c.py`
            # when user is using hydra find the absolute path
            path_lib = os.path.abspath

            # pull out the commands used to run the script and resolve the abs file path
            command = sys.argv
            try:
                full_path = path_lib(command[0])
            except Exception:
                full_path = os.path.abspath(command[0])

            command[0] = full_path
            # use the same python interpreter and actually running
            command = [sys.executable] + command
        else:  # Script called as `python -m a.b.c`
            command = [sys.executable, "-m", __main__.__spec__.name] + sys.argv[1:]

        for rank in range(1, WORLD_SIZE):
            env_copy = os.environ.copy()
            env_copy["LOCAL_RANK"] = f"{rank}"
            proc = subprocess.Popen(command, env=env_copy)
            self.interactive_ddp_procs.append(proc)
            # delay = np.random.uniform(1, 5, 1)[0]
            # sleep(delay)

    def setup_distributed(self):
        dist.init_process_group(backend="nccl", world_size=WORLD_SIZE, rank=ENV.local_rank())


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    tensors = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors, tensor)
    return torch.stack(tensors, dim=0)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2"
    os.environ['PL_GLOBAL_SEED'] = '99'

    plugin = DDP()
    if not ENV.creates_processes_externally():
        plugin.call_children_scripts()
    plugin.setup_distributed()

    # 创建 DDP 模型进行分布式训练
    torch.cuda.set_device(ENV.local_rank())

    accuracy = MyAccuracy(device=ENV.local_rank())

    # torchmetrics/metric.py Metric 继承了 torch.nn.Module，且 metric 是 model 的一个实例的参数，所以在执行 model.to(device) 会执行 metric._apply
    # 里面有 if isinstance(current_val, Tensor): setattr(this, key, fn(current_val))
    # 把数据转为 device 的数据
    accuracy.to()
    if ENV.local_rank() == 0:
        preds = torch.tensor([0, 1, 0]).to(ENV.local_rank())
        target = torch.tensor([1, 1, 0]).to(ENV.local_rank())

    else:
        preds = torch.tensor([3, 1, 3]).to(ENV.local_rank())
        target = torch.tensor([0, 1, 0]).to(ENV.local_rank())

    acc = accuracy.execute_update(preds, target)
    acc_expected = torch.eq(preds, target).sum() / preds.shape[0]
    print(f'RANK: {ENV.local_rank()} acc: {acc} acc_expected: {acc_expected}')

    acc = accuracy.execute_compute()
    print(f'RANK: {ENV.local_rank()} acc: {acc} acc_expected: {3 / 6}')

    for p in plugin.interactive_ddp_procs:
        p.wait()
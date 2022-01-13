import os
import subprocess
import sys
import __main__
from typing import List

import torch.distributed as dist
import torch.utils.data
import torch

from ai_metrics.metrics.torch_metrics import Accuracy

WORLD_SIZE = 2
torch.set_printoptions(linewidth=200)


class Environment:
    """
    每个 node 都有 num_processes 个 process，一共有self.world_size 个 process
    """
    def __init__(self):
        self._global_rank: int = 0
        self._world_size: int = 1

    @property
    def creates_processes_externally(self) -> bool:
        """Returns whether the cluster creates the processes or not.
        If at least :code:`LOCAL_RANK` is available as environment variable, Lightning assumes the user acts as the
        process launcher/job scheduler and Lightning will not launch new processes.
        """
        return "LOCAL_RANK" in os.environ

    def world_size(self) -> int:
        """The number of processes across all devices and nodes."""
        return self._world_size

    def set_world_size(self, size: int) -> None:
        self._world_size = size

    def local_rank(self) -> int:
        """The rank (index) of the currently running process inside of the current node."""
        return int(os.environ.get("LOCAL_RANK", 0))

    def global_rank(self) -> int:
        """The rank (index) of the currently running process across all nodes and devices."""
        return self._global_rank

    def set_global_rank(self, rank: int) -> None:
        self._global_rank = rank


class DDP:
    def __init__(self, parallel_devices: List[torch.device], environment: Environment):
        """
        每个 node 都有 num_processes 个 process
        :param parallel_devices:
        :param environment:
        """
        self.interactive_ddp_procs = []
        self.parallel_devices = parallel_devices
        self.environment = environment
        self.num_processes = len(self.parallel_devices)
        self.num_nodes = 1
        self.node_rank = 0

    def setup_environment(self) -> None:
        if not self.environment.creates_processes_externally:
            self.call_children_scripts()

        self.setup_distributed()

    def call_children_scripts(self) -> None:
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

        for rank in range(1, self.num_processes):
            env_copy = os.environ.copy()
            env_copy["LOCAL_RANK"] = f"{rank}"
            proc = subprocess.Popen(command, env=env_copy)
            self.interactive_ddp_procs.append(proc)
            # delay = np.random.uniform(1, 5, 1)[0]
            # sleep(delay)

    def setup_distributed(self) -> None:
        self.environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
        self.environment.set_world_size(self.num_nodes * self.num_processes)

        global_rank = self.environment.global_rank()
        world_size = self.environment.world_size()
        dist.init_process_group(backend="nccl", world_size=world_size, rank=global_rank)

    @property
    def local_rank(self) -> int:
        return self.environment.local_rank()

    @property
    def root_device(self) -> torch.device:
        return self.parallel_devices[self.local_rank]


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    tensors = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors, tensor)
    return torch.stack(tensors, dim=0)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

    gpu_ids = [0, 1]
    devices = [torch.device("cuda", i) for i in gpu_ids]
    environment = Environment()

    plugin = DDP(parallel_devices=devices, environment=environment)

    plugin.setup_environment()

    # 创建 DDP 模型进行分布式训练
    torch.cuda.set_device(plugin.environment.local_rank())

    accuracy = Accuracy(need_explicit_to=False)

    # need_explicit_to 为 False，不需要显式 metric.to(device)
    # accuracy.to(plugin.root_device)

    # shape: [n=2, batch_size] 表示对于每个设备，每个 epoch 有 n/2 次 metric 使用，每次都是一个 batch_size（一共两个设备）
    predict = torch.tensor([
        [1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 0, 0, 1, 2, 2, 0, 0, 1, 0, 0, 1, 1, 1, 2, 2, 0, 2, 1, 2],
        [1, 2, 0, 2, 1, 2, 0, 1, 2, 0, 0, 0, 1, 0, 1, 0, 1, 0, 2, 2, 1, 2, 0, 0, 1, 0, 2, 2, 2, 1, 1]])
    target = torch.tensor([
        [1, 0, 1, 2, 1, 2, 1, 0, 0, 1, 0, 1, 2, 1, 2, 0, 1, 0, 0, 0, 2, 0, 0, 1, 1, 0, 1, 0, 1, 2, 2],
        [1, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 0, 2, 2, 1, 2, 1, 0, 2, 1, 1, 2, 2, 0]])

    using_predict = predict[environment.local_rank()].to(plugin.root_device)
    using_target = target[environment.local_rank()].to(plugin.root_device)

    acc = accuracy.execute_evaluate(using_predict, using_target)
    acc_expected = torch.eq(using_predict, using_target).sum().to(torch.float64) / using_predict.shape[0]
    print(f'RANK: {plugin.local_rank} acc: {acc} acc_expected: {acc_expected}')

    acc = accuracy.execute_get_metric()
    acc_expected = torch.eq(predict, target).sum().to(torch.float64) / target.numel()
    print(f'RANK: {plugin.local_rank} acc: {acc} acc_expected: {acc_expected}')


if __name__ == "__main__":
    main()

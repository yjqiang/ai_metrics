from itertools import product
from typing import List, Tuple, Any

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


def cases_for_function(function: Any) -> List[List[Tuple[str, Any]]]:
    """
    https://gist.github.com/yjqiang/804b1210945e24838a755c239cc5a998
    返回所有取值情况（与 pytest 一致）
    :param function: 用 pytest.mark.parametrize 包裹的待测试的函数
    :return: [[(case0_arg_name0, case0_arg_value0), (case0_arg_name1, case0_arg_value1), ...], [case1], [case2], ...]
    """
    list_cases: List[List[Tuple[str, Any]]] = []  # [[(case0_arg_name0, case0_arg_value0), (case0_arg_name1, case0_arg_value1), ...], [case1], [case2], ...]
    # for mark in function.pytestmark 每个 mark 都是一个用户定义的 pytest.mark.parametrize
    # mark.args[0] 是参数名 arg_name，mark.args[1] 是该参数的所有可能（参数名可能是 'x, y' 形式）
    # 实现了排列组合生成所有 case，与最终 list_cases 基本一致。但这里还不够，有的 case(i)_arg_name(j) 是 'x, y' 形式的
    list_cases_ = product(*[[(mark.args[0], value) for value in mark.args[1]] for mark in function.pytestmark])
    for case_ in list_cases_:  # [(case_arg_name0, case_arg_value0), (case_arg_name1, case_arg_value1), ...]；但这里还不够，有的 arg_name(j) 是 'x, y' 形式的
        case: List[Tuple[str, Any]] = []  # case 是对 case_ 的改进，即把其中 'x, y' 的形式分离开来
        for arg_name_, arg_value_ in case_:
            arg_name_ = [x.strip() for x in arg_name_.split(",") if x.strip()]
            if len(arg_name_) > 1:  # eg: parametrize("a, b", [(-1, -2), (-3, -3)])
                args_values = arg_value_  # 形式 [args0, args1, ...]，每个 args 的 len 都是 len(args_names)
                args_names = arg_name_
                for arg_value, args_name in zip(args_names, args_values):
                    case.append((arg_value, args_name))
            else:  # eg: @pytest.mark.parametrize("xx", [0, 1])
                arg_name = arg_name_[0]
                arg_value = arg_value_
                case.append((arg_name, arg_value))
        list_cases.append(case)
    return list_cases


def cases_for_method(method: Any) -> List[List[Tuple[str, Any]]]:
    """
    https://gist.github.com/yjqiang/804b1210945e24838a755c239cc5a998
    返回所有取值情况（与 pytest 一致）
    :param method: 用 pytest.mark.parametrize 包裹的待测试的方法（可以继承）
    :return: [[(case0_arg_name0, case0_arg_value0), (case0_arg_name1, case0_arg_value1), ...], [case1], [case2], ...]
    """
    # [[method_case0, class_case0], [method_case0, class_case1], ...]
    # 所有的 case 都是 [(case_arg_name0, case0_arg_value0), (case_arg_name1, case0_arg_value1), ...] 形式
    product_result = product(cases_for_function(method), cases_for_function(type(method.__self__)))
    return [[*method_case, *class_case] for method_case, class_case in product_result]


def pytest_function(function: Any) -> None:
    list_cases = cases_for_function(function)
    for case in list_cases:
        result = function(**{arg_name: arg_value for arg_name, arg_value in case})
        print(f'{function.__name__}({", ".join(f"{arg_name}={arg_value}" for arg_name, arg_value in case)}): result={result}')


def pytest_method(method: Any) -> None:
    list_cases = cases_for_method(method)
    for case in list_cases:
        result = method(**{arg_name: arg_value for arg_name, arg_value in case})
        print(f'{method.__name__}({", ".join(f"{arg_name}={arg_value}" for arg_name, arg_value in case)}): result={result}')

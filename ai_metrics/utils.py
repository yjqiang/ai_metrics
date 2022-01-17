from typing import Any, Callable, Union, Tuple, Type
from collections.abc import Mapping, Sequence
import dataclasses


def is_mapping(instance: Any) -> bool:
    """
    data 是否为 Mapping
    https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
    A mapping object maps hashable values to arbitrary objects. Mappings are mutable objects. There is currently only one standard mapping type, the dictionary.
    但可以自定义 mapping object 呀，例如 https://pypi.org/project/multidict/
    isinstance(multidict.CIMultiDict(key='val'), dict) 为 False，但 isinstance(multidict.CIMultiDict(key='val'), Mapping) 为 True

    :param instance:
    :return:
    """
    return isinstance(instance, Mapping)


def is_namedtuple(instance: Any) -> bool:
    """
    如果 data 为 namedtuple，这里怎么判定应该没有定论
    https://bugs.python.org/issue7796
    https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple

    :param instance:
    :return:
    """
    return isinstance(instance, tuple) and hasattr(instance, '_asdict') and hasattr(instance, '_fields')


def is_sequence(instance: Any) -> bool:
    """
    如果 data 为 Sequence
    https://stackoverflow.com/a/2937122
    https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range
    There are three basic sequence types: lists, tuples, and range objects.

    :param instance:
    :return:
    """
    return isinstance(instance, Sequence) and not isinstance(instance, str)


def is_dataclass(instance: Any) -> bool:
    """
    如果 data 为 dataclass
    https://docs.python.org/3/library/dataclasses.html#dataclasses.is_dataclass

    :param instance:
    :return:
    """

    return dataclasses.is_dataclass(instance) and not isinstance(instance, type)


def apply_to_collection(
    data: Any,
    wanted_type: Union[Type, Tuple[Type, ...]],
    function: Callable,
    **kwargs
) -> Any:
    """
    递归查看 data 里面的所有元素，并把符合 wanted_type 的元素全部执行 function(data, **kwargs)
    请注意：本函数不一定执行了复制操作
    :param data:
    :param wanted_type:
    :param function:
    :param kwargs: 执行 function 时候的补充参数
    :return:
    """
    data_type = type(data)

    if isinstance(data, wanted_type):
        return function(data, **kwargs)

    # 如果 data 为 Mapping
    if is_mapping(data):
        return data_type({key: apply_to_collection(value, wanted_type, function, **kwargs) for key, value in data.items()})

    # 如果 data 为 namedtuple
    if is_namedtuple(data):
        return data_type(*(apply_to_collection(element, wanted_type, function, **kwargs) for element in data))

    # 如果 data 为 Sequence
    if is_sequence(data):
        return data_type([apply_to_collection(element, wanted_type, function, **kwargs) for element in data])

    # 如果 data 为 dataclass
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        out_dict = {}
        for field in dataclasses.fields(data):
            # https://docs.python.org/3/library/dataclasses.html#dataclasses.field
            # init: If true (the default), this field is included as a parameter to the generated __init__() method.
            if field.init:
                value = apply_to_collection(getattr(data, field.name), wanted_type, function, **kwargs)
                out_dict[field.name] = value
        return data_type(**out_dict)

    return data


def check_collection(
    data: Any,
    wanted_type: Union[Type, Tuple[Type, ...]],
    function: Callable,
    **kwargs
) -> Any:
    """
    递归查看 data 里面的所有元素，并把符合 wanted_type 的元素执行 function(data, **kwargs)，并把结果返回（仅对第一个符合 wanted_type 的元素执行）
    请注意：这里假设所有符合 wanted_type 的元素，执行 function 后都是一样的结果，且非 None；function 不要返回 None

    :param data:
    :param wanted_type:
    :param function:
    :param kwargs: 执行 function 时候的补充参数
    :return:
    """
    if isinstance(data, wanted_type):
        return function(data, **kwargs)

    # 如果 data 为 Mapping
    if is_mapping(data):
        for value in data.values():
            result = check_collection(value, wanted_type, function, **kwargs)
            if result is not None:
                return result

    # 如果 data 为 namedtuple 或 Sequence
    elif is_namedtuple(data) or is_sequence(data):
        for element in data:
            result = check_collection(element, wanted_type, function, **kwargs)
            if result is not None:
                return result

    # 如果 data 为 dataclass
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        for field in dataclasses.fields(data):
            # https://docs.python.org/3/library/dataclasses.html#dataclasses.field
            # init: If true (the default), this field is included as a parameter to the generated __init__() method.
            if field.init:
                result = check_collection(getattr(data, field.name), wanted_type, function, **kwargs)
                if result is not None:
                    return result

    return None

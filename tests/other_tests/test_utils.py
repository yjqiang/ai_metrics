import dataclasses
import numbers
from collections import namedtuple
from collections.abc import Mapping
from typing import List, Dict, Any, Iterator

import torch
import numpy as np

from ai_metrics import utils


def test_recursive_application_to_collection():
    Namedtuple = namedtuple('Namedtuple', ['namedtuple_field0', 'namedtuple_field1'])

    @dataclasses.dataclass
    class Dataclass0:
        dataclass0_field0: torch.Tensor
        dataclass0_field1: np.ndarray

    @dataclasses.dataclass
    class Dataclass1:
        dataclass1_field0: List[str]
        dataclass1_field1: Dataclass0
        dataclass1_field2: torch.Tensor
        dataclass1_field3_constant: int = dataclasses.field(init=False)

        def __post_init__(self):
            self.dataclass1_field3_constant = 7

    class MyMapping(Mapping):
        def __init__(self, __arg: Dict[str, Any]) -> None:
            self._dict = __arg

        def __len__(self) -> int:
            return len(self._dict)

        def __getitem__(self, key: str) -> Any:
            return self._dict[f'{key}']

        def __iter__(self) -> Iterator[str]:
            return iter(self._dict)

    to_reduce = {
        'a': torch.tensor([1.0]),  # Tensor
        'b': [torch.tensor([2.0]), np.array([4.0, 5.0, 6.0]), 3.0],  # list
        'c': (torch.tensor([100.0]),),  # tuple
        'd': Namedtuple(namedtuple_field0=5.0, namedtuple_field1=torch.tensor([5.0])),  # namedtuple
        'e': np.array([10.0]),  # np.ndarray
        'f': 'this_is_a_dummy_str',  # string
        'g': 12.0,  # number
        'h': Dataclass0(dataclass0_field0=torch.tensor([1.0, 2.0, 3.0]), dataclass0_field1=np.array([4.0, 5.0, 6.0])),  # dataclass
        'i': Dataclass1(
            dataclass1_field0=['i-1', 'i-2', 'i-3'],
            dataclass1_field1=Dataclass0(dataclass0_field0=torch.tensor([1.0, 2.0, 3.0]), dataclass0_field1=np.array([4.0, 5.0, 6.0])),
            dataclass1_field2=torch.tensor([7.0, 8.0, 9.0]),
        ),  # nested dataclass
        'j': MyMapping({'a': 2, 'b': 1}),  # mapping
    }

    expected_results = [
        # 约束 (torch.Tensor, numbers.Number, np.ndarray)
        {
            'a': torch.tensor([2.0]),
            'b': [torch.tensor([4.0]), np.array([8.0, 10.0, 12.0]), 6.0],
            'c': (torch.tensor([200.0]),),
            'd': Namedtuple(namedtuple_field0=10.0, namedtuple_field1=torch.tensor([10.0])),
            'e': np.array([20.0]),
            'f': 'this_is_a_dummy_str',
            'g': 24.0,
            'h': Dataclass0(dataclass0_field0=torch.tensor([2.0, 4.0, 6.0]), dataclass0_field1=np.array([8.0, 10.0, 12.0])),
            'i': Dataclass1(
                dataclass1_field0=['i-1', 'i-2', 'i-3'],
                dataclass1_field1=Dataclass0(dataclass0_field0=torch.tensor([2.0, 4.0, 6.0]), dataclass0_field1=np.array([8.0, 10.0, 12.0])),
                dataclass1_field2=torch.tensor([14.0, 16.0, 18.0]),
            ),
            'j': MyMapping({'a': 4, 'b': 2})
        },
        # 约束 (torch.Tensor, np.ndarray)
        {
            'a': torch.tensor([2.0]),
            'b': [torch.tensor([4.0]), np.array([8.0, 10.0, 12.0]), 3.0],
            'c': (torch.tensor([200.0]),),
            'd': Namedtuple(namedtuple_field0=5.0, namedtuple_field1=torch.tensor([10.0])),
            'e': np.array([20.0]),
            'f': 'this_is_a_dummy_str',
            'g': 12.0,
            'h': Dataclass0(dataclass0_field0=torch.tensor([2.0, 4.0, 6.0]), dataclass0_field1=np.array([8.0, 10.0, 12.0])),
            'i': Dataclass1(
                dataclass1_field0=['i-1', 'i-2', 'i-3'],
                dataclass1_field1=Dataclass0(dataclass0_field0=torch.tensor([2.0, 4.0, 6.0]), dataclass0_field1=np.array([8.0, 10.0, 12.0])),
                dataclass1_field2=torch.tensor([14.0, 16.0, 18.0]),
            ),
            'j': MyMapping({'a': 2, 'b': 1})
        },
        # 约束 torch.Tensor
        {
            'a': torch.tensor([2.0]),
            'b': [torch.tensor([4.0]), np.array([4.0, 5.0, 6.0]), 3.0],
            'c': (torch.tensor([200.0]),),
            'd': Namedtuple(namedtuple_field0=5.0, namedtuple_field1=torch.tensor([10.0])),
            'e': np.array([10.0]),
            'f': 'this_is_a_dummy_str',
            'g': 12.0,
            'h': Dataclass0(dataclass0_field0=torch.tensor([2.0, 4.0, 6.0]), dataclass0_field1=np.array([4.0, 5.0, 6.0])),
            'i': Dataclass1(
                dataclass1_field0=['i-1', 'i-2', 'i-3'],
                dataclass1_field1=Dataclass0(dataclass0_field0=torch.tensor([2.0, 4.0, 6.0]), dataclass0_field1=np.array([4.0, 5.0, 6.0])),
                dataclass1_field2=torch.tensor([14.0, 16.0, 18.0]),
            ),
            'j': MyMapping({'a': 2, 'b': 1})
        },
    ]

    for reduced_result, expected_result in zip(
            [
                utils.apply_to_collection(to_reduce, (torch.Tensor, numbers.Number, np.ndarray), lambda value: value * 2),
                utils.apply_to_collection(to_reduce, (torch.Tensor, np.ndarray), lambda value: value * 2),
                utils.apply_to_collection(to_reduce, torch.Tensor, lambda value: value * 2),
            ],
            expected_results
    ):

        # print('>>>>>>>', reduced_result)
        # print('>>>>>>>', expected_result)
        # print('-' * 100)
        assert isinstance(reduced_result, dict)
        assert all(isinstance(reduced_result[k], type(expected_result[k])) for k in to_reduce)

        # a Tensor
        assert isinstance(reduced_result['a'], torch.Tensor)
        assert torch.allclose(reduced_result['a'], expected_result['a'])

        # b list
        assert isinstance(reduced_result['b'], list)
        for x, y in zip(reduced_result['b'], expected_result['b']):
            if isinstance(x, torch.Tensor):
                assert torch.allclose(x, y)
            elif isinstance(x, np.ndarray):
                assert np.allclose(x, y)
            else:
                assert x == y

        # c tuple
        assert isinstance(reduced_result['c'], tuple)
        assert all(torch.allclose(x, y) for x, y in zip(reduced_result['c'], expected_result['c']))

        # d namedtuple
        assert isinstance(reduced_result['d'], Namedtuple)
        assert reduced_result['d'].namedtuple_field0 == expected_result['d'].namedtuple_field0
        assert torch.allclose(reduced_result['d'].namedtuple_field1, expected_result['d'].namedtuple_field1)

        # e np.ndarray
        assert isinstance(reduced_result['e'], np.ndarray)
        assert reduced_result['e'] == expected_result['e']

        # f string
        assert isinstance(reduced_result['f'], str)
        assert reduced_result['f'] == expected_result['f']

        # g number
        assert isinstance(reduced_result['g'], numbers.Number)
        assert reduced_result['g'] == expected_result['g']

        # h dataclass
        # https://docs.python.org/3/library/dataclasses.html#dataclasses.is_dataclass
        assert dataclasses.is_dataclass(reduced_result['h']) and not isinstance(reduced_result['h'], type)
        assert torch.allclose(reduced_result['h'].dataclass0_field0, expected_result['h'].dataclass0_field0)
        assert np.allclose(reduced_result['h'].dataclass0_field1, expected_result['h'].dataclass0_field1)

        # i nested dataclass
        # https://docs.python.org/3/library/dataclasses.html#dataclasses.is_dataclass
        assert dataclasses.is_dataclass(reduced_result['i']) and not isinstance(reduced_result['i'], type)
        assert dataclasses.is_dataclass(reduced_result['i'].dataclass1_field1) and not isinstance(reduced_result['i'].dataclass1_field1, type)
        assert reduced_result['i'].dataclass1_field0 == expected_result['i'].dataclass1_field0
        assert torch.allclose(reduced_result['i'].dataclass1_field1.dataclass0_field0, expected_result['i'].dataclass1_field1.dataclass0_field0)
        assert np.allclose(reduced_result['i'].dataclass1_field1.dataclass0_field1, expected_result['i'].dataclass1_field1.dataclass0_field1)
        assert torch.allclose(reduced_result['i'].dataclass1_field2, expected_result['i'].dataclass1_field2)
        assert reduced_result['i'].dataclass1_field3_constant == expected_result['i'].dataclass1_field3_constant

        # j mapping
        assert isinstance(reduced_result['j'], MyMapping)
        assert all(key0 == key1 and value0 == value1 for (key0, value0), (key1, value1) in zip(reduced_result['j'].items(), expected_result['j'].items()))

        # print('DONE')

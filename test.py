import subprocess
from typing import List, Union


def encode_gpu_indexes(gpu_indexes: Union[List, List[int]]) -> str:
    str_list = [str(idx) for idx in gpu_indexes]
    result = ",".join(str_list)
    return result


def decode_gpu_indexes(str_gpu_indexes: str) -> List[int]:
    split = str_gpu_indexes.split(",")
    return [int(idx) for idx in split]


int_list = [0]

to_str = encode_gpu_indexes(int_list)

print(to_str)

to_list = decode_gpu_indexes(to_str)

print(to_list)

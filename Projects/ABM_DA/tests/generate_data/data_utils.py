# Imports
from math import atan2
# Functions
def __check_var_iterables(var_iterables: tuple) -> None:
    for i in range(len(var_iterables)-1):
        assert len(var_iterables[i]) == len(var_iterables[i+1])


def wrap_up(var_iterables: tuple) -> list:
    __check_var_iterables(var_iterables)

    output = list()

    for i in range(len(var_iterables[0])):
        x = list()
        for var in var_iterables:
            x.append(var[i])
        x = tuple(x)
        output.append(x)

    return output


def get_angle(vector_tail, vector_head):
    x_diff = vector_head[0] - vector_tail[0]
    y_diff = vector_head[1] - vector_tail[1]
    angle = atan2(y_diff, x_diff)
    return angle

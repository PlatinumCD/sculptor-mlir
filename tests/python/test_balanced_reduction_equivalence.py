#!/usr/bin/env python3

import math
import random
import struct


def sequential_reduce(values, operation):
    result = values[0].copy()
    for value in values[1:]:
        result = operation(result, value)
    return result


def width_reduce(values, operation, width):
    base_size, extra_lanes = divmod(len(values), width)
    lane_results = []
    offset = 0
    for lane in range(width):
        lane_size = base_size + (lane < extra_lanes)
        lane_values = values[offset : offset + lane_size]
        offset += lane_size
        lane_results.append(sequential_reduce(lane_values, operation))
    return sequential_reduce(lane_results, operation)


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def elementwise(operation):
    return lambda lhs, rhs: [operation(x, y) for x, y in zip(lhs, rhs)]


def addf(lhs, rhs):
    return f32(lhs + rhs)


def check_reduction(fan_in, width, operation, exact):
    rng = random.Random(1000 + fan_in)
    values = [
        [f32(rng.uniform(-2.0, 2.0)) for _ in range(24)]
        for _ in range(fan_in)
    ]
    expected = sequential_reduce(values, operation)
    actual = width_reduce(values, operation, width)
    if exact:
        assert actual == expected
    else:
        assert all(
            math.isclose(lhs, rhs, rel_tol=1.0e-6, abs_tol=1.0e-6)
            for lhs, rhs in zip(actual, expected)
        )


def main():
    for fan_in, width in ((3, 2), (4, 2), (5, 2), (32, 4)):
        check_reduction(fan_in, width, elementwise(addf), exact=False)
        check_reduction(fan_in, width, elementwise(max), exact=True)
        check_reduction(fan_in, width, elementwise(min), exact=True)
    print("width-controlled add, max, and min reductions match references")


if __name__ == "__main__":
    main()

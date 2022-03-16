import math
import random
from typing import List, Tuple, Dict

import numpy as np


def is_calculated(array: List[int], solves: Dict[int, Tuple[int, bool]]):
    for var in array:
        if not solves[var][1]:
            return False
    return True


def is_solved(solves: Dict[int, Tuple[int, bool]]):
    for v in solves.values():
        if not v[1]:
            return False
    return True


def gauss(sorted_matrix):
    sorted_matrix_size = len(sorted_matrix)
    null_vector = [0] * len(sorted_matrix[0])
    sole = {}

    i = 0
    while i < sorted_matrix_size:
        k = 0
        sorted_matrix = sorted(np.unique(sorted_matrix, axis=0).tolist(), key=lambda x: x, reverse=True)

        while null_vector in sorted_matrix:
            sorted_matrix.remove(null_vector)

        sorted_matrix_size = len(sorted_matrix)
        if i >= sorted_matrix_size:
            break

        for k in range(i, len(null_vector)):
            if sorted_matrix[i][k] == 1:
                sole[k] = []
                for w in range(k + 1, len(null_vector)):
                    if sorted_matrix[i][w] == 1:
                        sole[k].append(w)
                break
        for j in range(i + 1, len(sorted_matrix)):
            if sorted_matrix[j][k] == 1:
                sorted_matrix[j] = np.abs(np.subtract(sorted_matrix[j], sorted_matrix[i])).tolist()
        i += 1

    while [0] * (len(sorted_matrix[0])) in sorted_matrix:
        sorted_matrix.remove([0] * (len(sorted_matrix) + 1))

    sole = {k: v for k, v in sorted(sole.items(), key=lambda item: len(item[1]))}
    sole_solve = {k: (0, False) for k in range(len(null_vector))}

    for k, v in sole.items():
        if not v:
            sole_solve[k] = (0, True)

    for var in range(len(null_vector)):
        if var not in sole.keys():
            sole_solve[var] = (1, True)

    while not is_solved(sole_solve):
        for k, v in sole.items():
            if is_calculated(v, sole_solve):
                value = 0
                for var in v:
                    value += sole_solve[var][0]
                sole_solve[k] = (value % 2, True)

    sole_solve = {k: sole_solve[k] for k in sorted(sole_solve)}

    coefficients = [v[0] for v in sole_solve.values()]

    return coefficients


def is_smooth(x: int, factors: List[int]):
    for factor in factors:
        while x % factor == 0:
            x //= factor
    if x != 1:
        return False
    return True


def get_degrees(x: int, factors: List[int]):
    degrees = []
    for factor in factors:
        degrees.append(0)
        while x % factor == 0:
            degrees[-1] += 1
            x //= factor
    return degrees


def eratosthenes(n: int):
    sieve = list(range(n + 1))

    sieve[1] = 0

    for i in range(2, n):
        if sieve[i] != 0:
            j = i + i
            while j <= n:
                sieve[j] = 0
                j = j + i

    return [x for x in sieve if x != 0]


def convert_to_epsilon_vectors(matrix: List[List[int]]):
    epsilon_matrix = matrix.copy()
    return list(map(lambda x: list(map(lambda y: y % 2, x)), epsilon_matrix))


def dikson(n: int) -> None:
    L = math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))
    print(f'{L = }')
    M = math.sqrt(L)
    factors = eratosthenes(int(M))

    while True:
        values = []
        degrees = []

        while len(values) < len(factors) + 1:
            m = random.choice(range(math.ceil(math.sqrt(n)), n))
            if m in values:
                continue
            m_squared = (m ** 2) % n
            if is_smooth(m_squared, factors):
                values.append(m)
                degrees.append(get_degrees(m_squared, factors))

        epsilon_vectors = convert_to_epsilon_vectors(degrees)
        if not np.any(epsilon_vectors):
            continue
        coefficients = gauss(np.transpose(epsilon_vectors).tolist())

        if coefficients == len(coefficients) * [0]:
            continue

        X = 1
        for index, value in enumerate(values):
            X *= (value ** coefficients[index]) % n
            X %= n

        Y = 1
        for j in range(len(factors)):
            degree = 0
            for i in range(len(coefficients)):
                degree += coefficients[i] * degrees[i][j]
            Y *= factors[j] ** (degree // 2)
            Y %= n

        if X == Y and X == n - Y and X + Y < n:
            continue

        u = math.gcd(X + Y, n)
        v = math.gcd(X - Y, n)
        if 1 < u < n and 1 < v < n and u * v == n:
            print(f'{n} = {u} * {v} = {u * v}')
            return
        else:
            continue


def main():
    dikson(3519)


if __name__ == '__main__':
    main()

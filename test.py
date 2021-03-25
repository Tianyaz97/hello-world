"""
USM 作业code
"""
import numpy as np
import math
from scipy import linalg
from sympy import *
from scipy.stats import norm
import matplotlib.pyplot as plt

"""
matrix1 = np.array([[100, 32, -48, 0, 0],
                    [32, 64, 51.2, 0, 0],
                    [-48, 51.2, 256, 0, 0],
                    [0, 0, 0, 225, 45],
                    [0, 0, 0, 45, 25]])

v = np.array([[-1, -1, -1, 1, 1]])

result = np.dot(v, matrix1)
result = np.dot(result, v.T)
print(math.sqrt(result))

L = linalg.cholesky(matrix1, lower=True)  # cholesky分解
print(L)
print(np.dot(L, L.T))


"""

# matrix1 = np.array([[0.022, 0.017, 0, 0, 0.012],
#                     [0.017, 0.086, 0, 0, -0.012],
#                     [0, 0, 0.039, 0.006, 0.012],
#                     [0, 0, 0.006, 0.01, -0.008],
#                     [0.012, -0.012, 0.012, -0.008, 0.039]])
# L = linalg.cholesky(matrix1, lower=True)  # cholesky分解
# print(L)
# print(linalg.inv(L))

# m1 = np.array([[0.149, 0, 0, 0, 0],
#                [0, 0.294, 0, 0, 0],
#                [0, 0, 0.198, 0, 0],
#                [0, 0, 0, 0.1, 0],
#                [0, 0, 0, 0, 0.198]])
#
# m2 = np.array([[1, 0.4, 0, 0, 0.4],
#                [0.4, 1, 0, 0, -0.2],
#                [0, 0, 1, 0.3, 0.3],
#                [0, 0, 0.3, 1, -0.4],
#                [0.4, -0.2, 0.3, -0.4, 1]])
# res = np.dot(m1, m2)
# res = np.dot(res, m1)
# print(res)

# mat = np.array([[0.6, 1],
#                 [1, 0]])
# mat = linalg.inv(mat)
# print(np.dot(mat, np.array([[1], [0]])))

# t = symbols('t')
# T = symbols('T')
# c1 = symbols('c1')
# c2 = symbols('c2')
# x = symbols('x')

# a = solve([x**2 + 0.6*x +1], [x])
# print(a)

# f = c1 * exp(-0.3 * t) * (cos(0.95 * t) + I*sin(0.95 * t)) + c2 * exp(-0.3 * t) * (cos(0.95 * t) - I*sin(0.95 * t))
# print(diff(f, t).subs({t: 0}))
# print(diff(f, t))

# ut = exp(-1.7 * T)
# ht = sin(0.95 * (t - T))*I
# st = ut * ht
# print(integrate(st, (T, 0, t)))

# print(integrate(exp(-1.7 * x) * cos(0.95 * (a - x)), (x, 0, a)))
# print(integrate(((-6/19)*sin(0.95*(a-x))),(x,0,a)))
# print(linsolve([x + a -1,x -a -(-6/19)*I],(x,a)))

a = symbols('a')
b = symbols('b')
t = symbols('t')
s = symbols('s')
T = symbols('T')

eq1 = exp(-s*t)

# print(integrate(eq1,(t, -1, 1)))
# res = solve([x**4 + 2*x**3 + x**2], [x])
# print(res)

p = symbols('p')
n = symbols('n')
m = symbols('m')

# f1 = 1 - (1 - p**n)**m
# print(diff(f1, n))

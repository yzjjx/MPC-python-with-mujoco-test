# import numpy as np
# from scipy.optimize import minimize
#
# def Prediction(x_k, E, H, N, p):
#     U0 = np.zeros(N * p)  # 一维初值
#
#     x_k = np.asarray(x_k)
#     E = np.asarray(E)
#     H = np.asarray(H)
#
#     def objective(U):
#         U = np.asarray(U)  #.reshape(-1)
#         return (U.T @ H @ U) + (2 * (E.T @ x_k)).T @ U
#
#     result = minimize(objective, U0)
#
#     u_k = result.x[0]   # 只要第一个元素用 result.x[0]
#     return u_k

#代码修改，加速版
import numpy as np
from numba import jit

# 通过JIT编译加速
@jit(nopython=True, cache=True)
# 定义加速函数，下面调用
def solve_fast_analytical(x_k, E, H, N, p, u_min, u_max):
    #求导后的结果
    rhs = -1*(E.T@x_k)
    # 直接解线性方程组
    # 求解 H * U = rhs
    U_opt = np.linalg.solve(H, rhs)
    # 取第一个控制量
    u_k = U_opt[0]

    if u_k > u_max:
        u_k = u_max
    elif u_k < u_min:
        u_k = u_min

    return u_k


def Prediction(x_k, E, H, N, p):
    # 数据类型转换，不用再考虑数据类型，速度更快
    # 确定x为一维数据
    x_arr = np.asarray(x_k).flatten().astype(np.float64)
    E_arr = np.asarray(E).astype(np.float64)
    H_arr = np.asarray(H).astype(np.float64)
    # 设定物理限幅
    u_min = -80.0
    u_max = 80.0
    # 调用Numba核进行加速
    return solve_fast_analytical(x_arr, E_arr, H_arr, N, p, u_min, u_max)
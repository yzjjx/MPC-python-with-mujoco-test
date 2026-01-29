# 基于mujoco的MPC控制方法尝试
## 简介
基于Dr.CAN视频，首先将代码转换到python的环境，因为之前建模是基于xy平面的，将xml文件转换为xy平面
## 相关链接
视频链接：【【MPC模型预测控制器】1_最优化控制和基本概念】 https://www.bilibili.com/video/BV1cL411n7KV/?share_source=copy_web&vd_source=8f5fa48049054344abfb574dffb160ed
倒立摆xml模型：https://github.com/google-deepmind/dm_control

## 文件解释
ori.xml：原始的倒立摆mujoco模型文件，该文件不适用于xy平面的倒立摆建模
cartpole.xml：更改后的mujoco模型文件，更换了坐标系、更换了相机位置、加入了传感器、增加了驱动器控制范围
math_MPC_test.py：倒立摆数学仿真，使用前向欧拉法离散化
model_MPC_test.py:倒立摆mujoco仿真
Prediction.py：最优化、预测部分函数，包含加速版本与不加速版本，不加速版本与Dr.CAN视频基本一致
MPC_Matrices.py:矩阵初始化部分函数
mujoco_ctrl.py:mujoco的一些学习代码实例（无用）
test.py：numpy的一些学习代码尝试（无用）
## 运行效果
运行math_MPC_test.py，不加速版本效果

![不加速版本](https://pic1.imgdb.cn/item/697b221b681ba7926cd990e8.png)

加速版本效果：

![加速版本](https://pic1.imgdb.cn/item/697b221b681ba7926cd990e7.png)

运行model_MPC_test.py，倒立摆能够正常收敛：

![加速版本](https://pic1.imgdb.cn/item/697b2383681ba7926cd990f7.png)

## 一些记录
运行加速库：**numba**
即时编译JIT（Just-in-time compilation）：JIT编译器会动态地将高级语言编写的代码转换为机器码，可以直接由计算机的处理器执行，这是在运行时完成的，也就是代码执行之前，因此称为“即时”。JIT针对特定的硬件和操作系统进行代码优化，可以使得python代码获得显著的性能提升。
运行加速库可以和numpy一起使用
简介网页：https://numba.pydata.org.cn/


**最小化函数minimize**使用方法：
https://geek-blogs.com/blog/minimize-python/

加速前的原始代码
```python
import numpy as np
from scipy.optimize import minimize

def Prediction(x_k, E, H, N, p):
    U0 = np.zeros(N * p)  # 一维初值

    x_k = np.asarray(x_k)
    E = np.asarray(E)
    H = np.asarray(H)

    def objective(U):
        U = np.asarray(U)  #.reshape(-1)
        return (U.T @ H @ U) + (2 * (E.T @ x_k)).T @ U

    result = minimize(objective, U0)

    u_k = result.x[0]   # 只要第一个元素用 result.x[0]
    return u_k
```

在加速之前，因为**之前的二次规划计算得出输入最大值远远小于输入限制条件**，因此可以对代价函数求导，转化为线性求最值，即Ax=B
加速版本：
```python
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
```




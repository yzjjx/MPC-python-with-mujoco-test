import numpy as np

# X_k=np.zeros((4,4))
# a = X_k.shape[0]
# X_k[:,0]=[3,2,4,5]
# print(a)
# print(X_k)

# N=3
# n=2
# p=1
# C=np.zeros(((N+1)*n,N*p))
# B=np.asmatrix([[0],[0]])
# tmp=np.eye(2)
# 矩阵上下拼接
# # M = np.vstack((np.eye(4), np.zeros((2, 4))))
# C[:, 0] = np.vstack((tmp@B, C[4 - n, range(0, -1 - p)]))

# 矩阵左右拼接
# A=[[1,2],[3,4]]
# B=[[5,6],[7,8]]
# C = np.hstack((A, B))

# 数组是否可以像矩阵一样相乘test---->数组不能像矩阵一样相乘
# A=[[1,2],[3,4]]
# B=[[2],[1]]
# A=np.asmatrix(A)
# B=np.asmatrix(B)
# C=A@B

# # 测试矩阵是否能够成组拼接、成组替换
# C=[[0,0,0],[0,0,0],[0,0,0],[0.5,0,0],[0.05,0.5,0],[1,0,0],[0.15,0.05,0],[2,1,0.5]]
# # 成组替换矩阵的第三行、四行，其中，第一列为2*2矩阵(A)
# A=[[2,3],[4,5]]
# C=np.asmatrix(C)
# A=np.asmatrix(A)
# print(C)
# #[0:n-1,0:n-1]
# # C=C[0:3,0:]
# #C[2:4,:]=np.hstack((A,C[0:2,0:1]))
# rows=[2,3]
# rows=np.array(rows)
# C[rows,:]=np.hstack((A,C[rows-2,0:1]))
# print(C)

# T=0.1
# A=np.asmatrix([[0,1,0,0],[0,3,4,0],[0,0,0,1],[0,5,6,0]])
# n=np.size(A,0)
# A=(np.eye(n)+T*A)
# print(A)
#
#错误
# A=[[0,1,3],[1,2,3],[3,4,5]]
# B=[[0,1,3],[1,2,3],[3,4,5]]
# C=A@B
# print(C)

# 测试一维数组
x_k=[[1],[2],[3]]
x_k_1 = np.asarray(x_k)
x_k = np.asarray(x_k).reshape(-1)
print(x_k_1)
print(x_k)
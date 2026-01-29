import numpy as np
import matplotlib.pyplot as plt
import MPC_Matrices
import Prediction
import scipy.signal
from scipy.linalg import solve_discrete_are

M = 1.0       # mass of the cart in kg
m = 0.1       # mass of the pole in kg
l = 0.55      # length of the pole in meters
J = 0.0084    # moment of inertia of the pole in kg*m^2
b = 0.0005    # damping coefficient
g = 9.81      # gravitational acceleration in m/s^2

T=0.01

Q_d=(M+m)*(m*l**2+J)
a_1=-b*(m*l**2+J)/(Q_d-m**2*l**2)
a_2=-m**2*l**2*g/(Q_d-m**2*l**2)
a_3 = m*l*b/(Q_d-m**2*l**2)
a_4 = m*g*l*(M+m)/(Q_d-m**2*l**2)

b_1=(m*l**2+J)/(Q_d-m**2*l**2)
b_2=-m*l/(Q_d-m**2*l**2)

A=np.asmatrix([[0,1,0,0],[0,a_1,a_2,0],[0,0,0,1],[0,a_3,a_4,0]])
n=np.size(A,0)
A=(np.eye(n)+T*A)
A=np.asmatrix(A)

B=np.asmatrix([[0],[b_1],[0],[b_2]])
p=np.size(B,1)
B=T*B
B=np.asmatrix(B)

C = np.eye(n)          # (n,n)
D = np.zeros((n, p))

# A, B, C, D, T = scipy.signal.cont2discrete((A, B, C, D), T, method="zoh")

Q=np.asmatrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
R=np.asmatrix([0.01])

k_steps=2000

X_k=np.zeros((n,k_steps+1))

X_k[:,0]=[0,0,0.05,0]

# # A: (n,n), B: (n,m), Qx: (n,n), R: (m,m)
# P = solve_discrete_are(A, B, X_k, R)
F =1*np.eye(n)
F=np.asmatrix(F)

U_k=np.zeros((p,k_steps))

# 在发散的情况下增加预测步长，会有一些效果
N=100

[E,H]=MPC_Matrices.MPC_Matrices(A,B,Q,R,F,N)

H = np.asmatrix((H + H.T)/2)

for i in range(k_steps):
    U_k[:,i]=Prediction.Prediction(X_k[:,i],E,H,N,p)
    U_k=np.asmatrix(U_k)
    X_k=np.asmatrix(X_k)
    X_k[:,i+1]=A@X_k[:,i]+B@U_k[:,i]

plt.figure()

plt.subplot(2, 1, 1)
for i in range(X_k.shape[0]):
    plt.plot(np.asarray(X_k[i, :]).ravel(), label=f"x{i+1}")
plt.legend()

plt.subplot(2, 1, 2)
for i in range(U_k.shape[0]):
    plt.plot(np.asarray(U_k[i, :]).ravel(), label=f"u{i+1}")
plt.legend()

plt.show()

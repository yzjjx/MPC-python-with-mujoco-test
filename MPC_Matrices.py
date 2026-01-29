import numpy as np
import scipy.linalg as sl

def MPC_Matrices(A,B,Q,R,F,N):
    n=A.shape[0]
    p=B.shape[1]

    M=np.vstack((np.eye(n),np.zeros((N*n,n))))

    C=np.zeros(((N+1)*n,N*p))
    C=np.asmatrix(C)
    tmp=np.eye(n)

    tmp=np.asmatrix(tmp)
    B=np.asmatrix(B)
    A=np.asmatrix(A)

    for i in range(1,N+1):
        rows = i*n+np.array(range(n))
        C[rows,:]=np.hstack((tmp@B,C[rows-n,0:-p]))
        tmp=A@tmp
        M[rows,:]=tmp

    Q_bar=np.kron(np.eye(N),Q)
    Q_bar=sl.block_diag(Q_bar,F)
    R_bar=np.kron(np.eye(N),R)

    G=M.T@Q_bar@M
    E=M.T@Q_bar@C
    H=C.T@Q_bar@C+R_bar


    return E,H
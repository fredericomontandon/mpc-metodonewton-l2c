"""
TCC L2C - Curso de Calculo Numerico Turma 2

@author: Frederico Augusto Montandon Lima
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

#Modelo
a = 0.9
b = 0.1

#Referencia
T_ref = 50

#Horizonte MPC
N=10

#Peso do Controle
lambda_u = 0.1

#Simulação
steps = 50


def funcao_custo(u_seq, T0):
    T_pred = T0
    J = 0.0
    
    for k in range(N):
        
        T_pred = a*T_pred + b*u_seq[k]
        
        erro = T_pred - T_ref
        
        J += erro**2 + lambda_u*u_seq[k]**2
        
    return J


#Supondo que a temperatura incial é 20ºC
T0 = 20
u_seq = [80,80,80,80,80]

#Gradiente
def gradiente(u_seq,T0):
    eps = 1e-3
    
    n=len(u_seq)
    
    grad = np.zeros(n)
    
    J0 = funcao_custo(u_seq, T0)
    
    for i in range(n):
        
        u_temp = np.copy(u_seq)
        
        u_temp[i] += eps
        
        grad[i] = (funcao_custo(u_temp, T0)-J0)/eps
        
    return grad

#Hessiana

def hessiana(u_seq, T0):

    eps = 1e-4

    n = len(u_seq)

    H = np.zeros((n,n))

    J = funcao_custo(u_seq, T0)

    for i in range(n):

        for j in range(n):

            u_ij = np.copy(u_seq)
            u_i = np.copy(u_seq)
            u_j = np.copy(u_seq)

            u_ij[i] += eps
            u_ij[j] += eps

            u_i[i] += eps

            u_j[j] += eps

            J_ij = funcao_custo(u_ij, T0)
            J_i = funcao_custo(u_i, T0)
            J_j = funcao_custo(u_j, T0)

            H[i,j] = (
                J_ij
                - J_i
                - J_j
                + J
            ) / (eps**2)

    return H


def newton_optimizer(T0):

    u_seq = np.zeros(N)

    max_iter = 15

    tol = 1e-6

    for x in range(max_iter):

        grad = gradiente(u_seq, T0)

        H = hessiana(u_seq, T0)

        delta = np.linalg.solve(H, grad)

        u_seq = u_seq - delta

        u_seq = np.clip(u_seq, 0, 100)

        if np.linalg.norm(delta) < tol:
            break

    return u_seq

#Simulação

T = np.zeros(steps)

u_hist = np.zeros(steps)

T[0] = 20


for k in range(steps-1):

    u_opt = newton_optimizer(T[k])

    u = u_opt[0]

    u_hist[k] = u

    T[k+1] = a*T[k] + b*u


#Graficos
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(T,label='Temperatura')
plt.axhline(
    T_ref,
    color='red',
    linestyle='--',
    label='Referência'
)

plt.ylabel('Temperatura')
plt.grid()
plt.legend()
plt.subplot(2,1,2)
plt.plot(u_hist)
plt.ylabel('Controle')
plt.xlabel('Tempo')
plt.grid()
plt.tight_layout()
plt.show()


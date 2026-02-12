from performance_benchmarking import compute_time_elapsed

import sympy as sp
from itertools import product

half = sp.S.Half

η = sp.diag(1,-1,-1,-1) # Minkowski metric
δ = sp.KroneckerDelta
ε = sp.LeviCivita
zero_vec = sp.zeros(4,1)

def L(I, J):
    L_IJ = sp.zeros(4)
    for mu in range(4):
        for nu in range(4):
            L_IJ[mu, nu] = δ(mu, I) * η[nu, J] - δ(mu, J) * η[nu, I]
    return L_IJ

def T(I):
    return sp.Matrix([δ(mu,I) for mu in range(4)])

T0 = T(0); T = (T(1), T(2), T(3)); # Translations
J = (L(2,3), L(3,1), L(1,2))       # Rotations
K = (L(0,1), L(0,2), L(0,3))       # Boosts

def commutator(X, Y):
    Y_is_translation = (Y.shape==(4,1))
    if Y_is_translation:
        return X*Y
    return X*Y - Y*X

for i in range(3):
    for j in range(3):
        k = ({0,1,2} - {i,j}).pop()
        
        assert commutator(J[i], J[j]) ==  ε(i,j,k)*J[k],   ("[J_i, J_j]", i,j,k)
        assert commutator(J[i], K[j]) ==  ε(i,j,k)*K[k],   ("[J_i, K_j]", i,j,k)
        assert commutator(K[i], K[j]) == -ε(i,j,k)*J[k],   ("[K_i, K_j]", i,j,k)

        assert commutator(J[i], T[j]) == ε(i,j,k)*T[k],    ("[J_i, T_j]", i,j,k)
        assert commutator(K[i], T[j]) == -δ(i,j)*T0,       ("[K_i, T_j]", i,j,k)

    assert commutator(J[i], T0) == zero_vec,               ("[J_i, T_0]", i,0)
    assert commutator(K[i], T0) == -T[i],                  ("[K_i, T_0]", i,0)

compute_time_elapsed("Poincare commutation relations verified")
# sp.pprint(J[2]) # Print J_3

from performance_benchmarking import compute_time_elapsed

import sympy as sp
from itertools import product

half = sp.S.Half

η = sp.diag(1,-1,-1,-1) # Minkowski metric
δ = sp.KroneckerDelta
ε = sp.LeviCivita
zero_vec = sp.zeros(4,1)

def L(I, J): # Lorentz generators defined in equation 2.1.7
    L_IJ = sp.zeros(4)
    for mu in range(4):
        for nu in range(4):
            L_IJ[mu, nu] = δ(mu, I) * η[nu, J] - δ(mu, J) * η[nu, I]
    return L_IJ

def T(I): # Translation generators defined in text
    return sp.Matrix([δ(mu,I) for mu in range(4)])

T0 = T(0); T = (T(1), T(2), T(3)); # Translations
J = (L(2,3), L(3,1), L(1,2))       # Rotations
K = (L(0,1), L(0,2), L(0,3))       # Boosts

# Defined in equation 2.1.10
N_1 = {"+": K[0] + J[1], "-": K[0] - J[1]}
N_2 = {"+": K[1] - J[0], "-": K[1] + J[0]}

def commutator(X, Y):
    Y_is_translation = (Y.shape==(4,1))
    if Y_is_translation:
        return X*Y
    return X*Y - Y*X

# Verification of equations 2.1.9 and 2.1.11

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

assert commutator(N_1["+"], N_1["-"]) == -2*K[2]
assert commutator(N_2["+"], N_2["-"]) == -2*K[2]
assert commutator(N_1["+"], N_2["-"]) == -2*J[2]
assert commutator(N_1["-"], N_2["+"]) == -2*J[2]

for sign in ("+", "-"):
    sign_factor = -1 if sign == "-" else 1
    assert commutator(N_1[sign], N_2[sign]) == sp.zeros(4,4)
    assert commutator(N_1[sign], J[2]) == -N_2[sign]
    assert commutator(N_2[sign], J[2]) ==  N_1[sign]
    assert commutator(N_1[sign], K[2]) == sign_factor * N_1[sign]
    assert commutator(N_2[sign], K[2]) == sign_factor * N_2[sign]

    assert commutator(N_1[sign], T0 + sign_factor*T[2]) == zero_vec
    assert commutator(N_2[sign], T0 + sign_factor*T[2]) == zero_vec

compute_time_elapsed("Poincare commutation relations verified")

# Verifying the Lorentz generators are orthonormal, up to a sign,
# with respect to the inner product defined in 2.2.1

for i in range(3):
    for j in range(3):
        assert -half * sp.trace(J[i] * J[j]) == δ(i,j), (i,j)
        assert -half * sp.trace(K[i] * K[j]) == -δ(i,j), (i,j)
        assert -half * sp.trace(J[i] * K[j]) == 0, (i,j)

compute_time_elapsed("Orthonormality of Lorentz generators verified")

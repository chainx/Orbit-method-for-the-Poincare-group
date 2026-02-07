import sympy as sp
from itertools import product

half = sp.S.Half

η = sp.diag(1,-1,-1,-1)         # Minkowski metric
δ = sp.KroneckerDelta
ε = sp.LeviCivita

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

def check_commutation_relations():
    for i in range(3):
        for j in range(3):
            k = ({0,1,2} - {i,j}).pop()
            
            assert commutator(J[i], J[j]) ==  ε(i,j,k)*J[k],   ("[J, J]", i,j,k)
            assert commutator(J[i], K[j]) ==  ε(i,j,k)*K[k],   ("[J, K]", i,j,k)
            assert commutator(K[i], K[j]) == -ε(i,j,k)*J[k],   ("[K, K]", i,j,k)

            assert commutator(J[i], T[j]) == ε(i,j,k)*T[k],    ("[J, T]", i,j,k)
            assert commutator(K[i], T[j]) == -δ(i,j)*T0,       ("[K, T]", i,j,k)

        assert commutator(J[i], T0) == sp.zeros(4, 1),         ("[J, T]", i,0)
        assert commutator(K[i], T0) == -T[i],                  ("[K, T]", i,0)

def compute_pauli_lubanski_vector():
    m = sp.symbols("m", positive=True, real=True)
    s = sp.symbols("s", positive=True, real=True)
    p_m = m*T0
    M_s = s*J[2]

    W = sp.zeros(4,1)
    for mu in range(4):
        for nu, rho, sigma in product(range(4), repeat=3):
            W[mu] += half*ε(mu,nu,rho,sigma)*(η*p_m)[nu]*(η*M_s)[rho,sigma]

    sp.pprint(W)

# check_commutation_relations()
# sp.pprint(J[2])
compute_pauli_lubanski_vector()
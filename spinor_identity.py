from itertools import product
import sympy as sp

I = sp.I

# Pauli matrices
σ0 = sp.Matrix([[1, 0],
                [0, 1]])
σ1 = sp.Matrix([[0, 1],
                [1, 0]])
σ2 = sp.Matrix([[0, -I],
                [I,  0]])
σ3 = sp.Matrix([[1,  0],
                [0, -1]])
σ = {0:σ1, 1:σ2, 2:σ3}

ε = sp.Matrix([[0,1],[-1,0]])

fails = []
for A, B, C, Ad, Bd, Cd in product(range(2), repeat=6):

    lhs = 0
    for i, j, k in product(range(3), repeat=3):
        lhs += sp.LeviCivita(i,j,k)*σ[i][A,Ad]*σ[j][B,Bd]*σ[k][C,Cd]

    rhs = 2*I*( ε[B,C]*σ0[A,Cd]*ε[Ad,Bd] - ε[A,B]*σ0[C,Ad]*ε[Bd,Cd] )

    assert sp.simplify(lhs - rhs) == 0, (A,B,C,Ad,Bd,Cd, sp.simplify(lhs-rhs))

print('Success!')
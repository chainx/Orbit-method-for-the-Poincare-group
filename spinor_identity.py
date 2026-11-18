from itertools import product
import sympy as sp

I = sp.I

# Pauli matrices, with the identity for the time component.
σ0 = sp.eye(2)
σ1 = sp.Matrix([[0, 1],
                [1, 0]])
σ2 = sp.Matrix([[0, -I],
                [I,  0]])
σ3 = sp.Matrix([[1,  0],
                [0, -1]])

# Equation (A.8): with spacetime index and both spinor indices lowered, and the dotted index first, 
# the unnormalized matrices are σ_bar[μ][Ad, A].
σ_bar = (σ0, -σ1, -σ2, -σ3)

# The identity uses σ^μ_{A Ad}. Transpose to put A first, and raise the
# spacetime index with diag(1, -1, -1, -1).
σ = (σ_bar[0].T, -σ_bar[1].T, -σ_bar[2].T, -σ_bar[3].T)

ε = sp.Matrix([[0,  1],
               [-1, 0]])

for A, B, C, Ad, Bd, Cd in product(range(2), repeat=6):
    lhs = sum(
        sp.LeviCivita(i, j, k) * σ[i][A, Ad] * σ[j][B, Bd] * σ[k][C, Cd]
        for i, j, k in product(range(1, 4), repeat=3)
    )
    rhs = -2 * I * (
        ε[B, C] * σ[0][A, Cd] * ε[Ad, Bd]
        - ε[A, B] * σ[0][C, Ad] * ε[Bd, Cd]
    )

    difference = sp.simplify(lhs - rhs)
    assert difference == 0, (A, B, C, Ad, Bd, Cd, difference)

print('Success!')

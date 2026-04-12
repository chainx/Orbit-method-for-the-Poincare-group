from discrete_transformations import act_gauge_transfo, impose_constraints, m, p, ε, ψ as ψ_l
from performance_benchmarking import compute_time_elapsed

import sympy as sp; from sympy import I

"""
    Verifying the Dirac field obtained from Fourier transforming the momentum profile of the polarized sections
    (see equation 2.4.16) satisfies the Dirac equation in momentum space
"""

# (\boldsymbol{\sigma}_\mu)^{A\dotA} = (1, \sigma^i)
def sigma(p):
    return sp.Matrix([
        [p[0] + p[3],   p[1] - I*p[2]],
        [p[1] + I*p[2], p[0] - p[3]],
    ])

# (\bar{\boldsymbol{\sigma}}_\mu)_{\dotA A} = (1, -\sigma^i)
def sigma_bar(p):
    return sp.Matrix([
        [p[0] - p[3],   -p[1] + I*p[2]],
        [-p[1] - I*p[2], p[0] + p[3]],
    ])

def p_slash(p):
    return sp.Matrix(sp.BlockMatrix([
        [sp.zeros(2), sigma(p)],     
        [sigma_bar(p), sp.zeros(2)],
    ]))

ψ_r = (1/m) * sigma_bar(p) * ψ_l # Using the formula for the bottom two components in equation 2.4.16
                                 # The sqrt(2) cancels since we're using \boldsymbol{\sigma} instead of \sigma

Dirac_spinor = sp.Matrix.vstack(ψ_l, ψ_r) # Combining the pair of 2-spinors into a Dirac 4-spinor 

result = (p_slash(p) - m * sp.eye(4)) * Dirac_spinor # Dirac equation in momentum space
for n in range(4):
    component = impose_constraints(result[n])
    assert component == 0, (n,component)

compute_time_elapsed(f"Verified Dirac equation")

from performance_benchmarking import compute_time_elapsed

from itertools import permutations, product
import sympy as sp

I = sp.I
half = sp.S.Half

# Spinor epsilon
ε = sp.Matrix([[0, 1],[-1, 0]])

# Pauli matrices
σ0 = sp.Matrix([[1,0],[0,1]])
σ1 = sp.Matrix([[0,1],[1,0]])
σ2 = sp.Matrix([[0,-I],[I,0]])
σ3 = sp.Matrix([[1,0],[0,-1]])

# Infeld–van der Waerden symbols with downstairs indices (σ_μ)_{A\dotA} (defined in equation A.8)
barσ = [σ0, -σ1, -σ2, -σ3]
def σ(mu, A, dotA): 
    return sp.diag(-1, 1, 1, 1)[mu, mu] * barσ[mu][dotA, A] / sp.sqrt(2)

def Lorentz_LeviCivita(mu,nu,rho,sig):
    """
    Lorentzian Levi-Civita tensor with all four indices lowered.

    We use ε^{0123} = +1. Lowering all four indices in signature (+---) gives ε_{0123} = −1.
    """
    return -sp.LeviCivita(mu,nu,rho,sig)

# ---------------------------------------------------------
# Verification of Hodge dual identity in equation 2.2.11
# ---------------------------------------------------------

p0, p1, p2, p3 = sp.symbols("p0 p1 p2 p3", real=True)
W0, W1, W2, W3 = sp.symbols("W0 W1 W2 W3", real=True)
p = sp.Matrix([p0, p1, p2, p3])
W = sp.Matrix([W0, W1, W2, W3])

η = sp.diag(1,-1,-1,-1) # Minkowski metric
contraction = lambda u, v: (u.T * η * v)[0]

def LHS(p, W):
    S = sp.zeros(4, 4)
    for mu, nu, rho, sigma in product(range(4), repeat=4):
        S[mu, nu] += Lorentz_LeviCivita(mu, nu, rho, sigma) * p[rho] * W[sigma]
    v = sp.zeros(4, 1)
    for mu, nu, rho, sigma in product(range(4), repeat=4):
        # Using η to convert p^μ to p_μ
        v[mu] += half*sp.LeviCivita(mu,nu,rho,sigma)*(η*p)[nu]*S[rho, sigma]
    return sp.simplify(v)

def RHS(p, W):
    return contraction(p, p)*W - contraction(p, W)*p

diff = sp.simplify(LHS(p, W) - RHS(p, W))
assert diff == sp.zeros(4, 1), diff
compute_time_elapsed("*(p ∧ *(p ∧ W) == η(p,p)W - η(p,W)p verified")

# ---------------------------------------------------------------------
# Verification of spinor form of Levi-Civita symbol in equation A.27
# ---------------------------------------------------------------------

# Spinor Levi-Civita symbol in spinor form
def spinor_levi_civita(A,B,C,D, Ad,Bd,Cd,Dd):
    return I*( ε[A,C]*ε[B,D]*ε[Ad,Dd]*ε[Bd,Cd]
              -ε[A,D]*ε[B,C]*ε[Ad,Cd]*ε[Bd,Dd] )

# Define the spinorized Levi-Civita via contraction with σ's:
# ε_{ABCD \dotA\dotB\dotC\dotD} := ε_{μνρσ} σ^μ_{A \dotA} σ^ν_{B \dotB} σ^ρ_{C \dotC} σ^σ_{D \dotD}
def spinor_levi_civita_computed(A,B,C,D, Ad,Bd,Cd,Dd):
    levi_civita_component = 0
    for mu, nu, rho, sig in permutations(range(4), 4):
        levi_civita_component += Lorentz_LeviCivita(mu,nu,rho,sig) * σ(mu,A,Ad) * σ(nu,B,Bd) * σ(rho,C,Cd) * σ(sig,D,Dd)
    return sp.simplify(levi_civita_component)

# Verify equality componentwise
diffs = []
for A,B,C,D,Ad,Bd,Cd,Dd in product(range(2), repeat=8):
    diffs.append(sp.simplify(spinor_levi_civita_computed(A,B,C,D,Ad,Bd,Cd,Dd) - spinor_levi_civita(A,B,C,D,Ad,Bd,Cd,Dd)))

assert set(diffs) == {0}, diffs
compute_time_elapsed("Spinor form of Levi-Civita symbol verified")

# -----------------------------------------------------------------------------------------
# Verification of hodge star eigenvectors and eigenvalues (sentence below equation A.27)
# -----------------------------------------------------------------------------------------

# Arbitrary symmetric bispinor Φ_{(AB)}
α, β, γ = sp.symbols("α β γ")
Φ = sp.Matrix([[α, β],
               [β, γ]])

# Define the bivector term: F_{A\dotA B\dotB} = Φ_{(AB)} ε_{\dotA\dotB}
def F(A,Ad,B,Bd):
    return Φ[A,B]*ε[Ad,Bd]

# Bivector term with the spinor indices raised
def F_up(A,Ad,B,Bd):
    F_up_component = 0
    for C,Cd,D,Dd in product(range(2), repeat=4):
        F_up_component += ε[A,C]*ε[Ad,Cd]*ε[B,D]*ε[Bd,Dd]*F(C,Cd,D,Dd)
    return sp.simplify(F_up_component)

# Hodge dual: (*F)_{A\dotA B\dotB} = 1/2 ε_{ABCD \dotA\dotB\dotC\dotD} F^{C\dotC D\dotD}
def star_F(A,Ad,B,Bd):
    star_F_component = 0
    for C,Cd,D,Dd in product(range(2), repeat=4):
        star_F_component += sp.Rational(1,2) * spinor_levi_civita(A,B,C,D, Ad,Bd,Cd,Dd) * F_up(C,Cd,D,Dd)
    return sp.simplify(star_F_component)

# ratios:        contains eigenvalue λ from ratios (*F)/F across all nonzero components
# eigen_val_eqs: contains (*F) + i F components, which should be 0 (i.e,. *F = -i F)
ratios, eigen_val_eqs = [], []
nonzero_components = []
for A,Ad,B,Bd in product(range(2), repeat=4):
    f  = sp.simplify(F(A,Ad,B,Bd))
    sf = sp.simplify(star_F(A,Ad,B,Bd))
    if f != 0 or sf != 0:
        nonzero_components.append((A,Ad,B,Bd))
        ratios.append(sp.simplify(sf / f))
        eigen_val_eqs.append(sp.simplify(sf + I*f))

assert set(ratios)=={-I}, (ratios, nonzero_components)
assert set(eigen_val_eqs)=={0},   (eigen_val_eqs,  nonzero_components)

compute_time_elapsed("Verified Φ_{(AB)} ε_{\dotA\dotB} has eigenvalue -i")

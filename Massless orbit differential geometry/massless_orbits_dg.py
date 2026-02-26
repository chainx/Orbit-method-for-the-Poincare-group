import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from performance_benchmarking import compute_time_elapsed

from functools import lru_cache

import sympy as sp
from sympy.diffgeom import Manifold, Patch, CoordSystem

# -------------
# Definitions
# -------------

I = sp.I
half = sp.S.Half
sqrt2 = sp.sqrt(2)

dot = lambda u, v: sum(ui*vi for ui, vi in zip(u, v))

# Pauli matrices
σ0 = sp.Matrix([[1,0],[0,1]])
σ1 = sp.Matrix([[0,1],[1,0]])
σ2 = sp.Matrix([[0,-I],[I,0]])
σ3 = sp.Matrix([[1,0],[0,-1]])
# (σ_μ)^{A\dotA} and (σbar_μ)_{\dotA A} respectively
σ = [σ0, σ1, σ2, σ3]; σbar = [σ0, -σ1, -σ2, -σ3]

η = sp.diag(1,-1,-1,-1)         # Minkowski metric
ε = sp.Matrix([[0, 1],[-1, 0]]) # Symplectic form used to raise and lower spinor indices

def sigma(p0, p1, p2, p3):
    return (1/sqrt2) * sp.Matrix([
        [p0 + p3,     p1 - I*p2],
        [p1 + I*p2,   p0 - p3]
    ])

# ------------------------------
# Diffgeom coordinate symbols
# ------------------------------

π0, π1, πb0, πb1 = sp.symbols("π0 π1 πb0 πb1")
ω0, ω1, ωb0, ωb1 = sp.symbols("ω0 ω1 ωb0 ωb1")
q0, q1, q2, q3 = sp.symbols("q0 q1 q2 q3", real=True)

h = sp.symbols("h", positive=True, real=True)

ε_P = sp.symbols("ε_P", real=True)  # parity label, ε_P^2=1
ε_T = sp.symbols("ε_T", real=True)  # time label, ε_T^2=1

# --------------------------------------------------------------------------
# Diffgeom coordinates (we keep ε_P,ε_T as parameters, not coordinates)
# --------------------------------------------------------------------------

coords = [π0,π1,πb0,πb1, ω0,ω1,ωb0,ωb1, q0,q1,q2,q3]
coord_names = [coord.name for coord in coords]

M = Manifold("M", len(coords))
P = Patch("P", M)
C = CoordSystem("C", P, coords)

π0f,π1f,πb0f,πb1f, ω0f,ω1f,ωb0f,ωb1f, q0f,q1f,q2f,q3f = C.coord_functions()
e_π0,e_π1,e_πb0,e_πb1, e_ω0,e_ω1,e_ωb0,e_ωb1, e_q0,e_q1,e_q2,e_q3 = C.base_vectors()
dπ0,dπ1,dπb0,dπb1, dω0,dω1,dωb0,dωb1, dq0, dq1, dq2, dq3          = C.base_oneforms()

sym2bv = dict(zip(coord_names, C.base_vectors()))
sym2bf = dict(zip(coord_names, C.base_oneforms()))
sym2cf = dict(zip(coord_names, C.coord_functions()))

e_π, e_πb = sp.Matrix([e_π0,e_π1]), sp.Matrix([e_πb0,e_πb1])
e_ω, e_ωb = sp.Matrix([e_ω0,e_ω1]), sp.Matrix([e_ωb0,e_ωb1])
e_q  = sp.Matrix([e_q0,e_q1,e_q2,e_q3])

dπ, dπb = sp.Matrix([dπ0,dπ1]), sp.Matrix([dπb0,dπb1])
dω, dωb = sp.Matrix([dω0,dω1]), sp.Matrix([dωb0,dωb1])
dq = sp.Matrix([dq0, dq1, dq2, dq3])

π, πb = sp.Matrix([π0f, π1f]), sp.Matrix([πb0f, πb1f])
ω, ωb = sp.Matrix([ω0f, ω1f]), sp.Matrix([ωb0f, ωb1f])

# ----------------------------------------
# Groebner ideal for “mod constraints”
# ----------------------------------------

Π0,Π1,ΠB0,ΠB1, Ω0,Ω1,ΩB0,ΩB1, Q0,Q1,Q2,Q3, EP,ET = sp.symbols(
    "Π0 Π1 ΠB0 ΠB1 Ω0 Ω1 ΩB0 ΩB1 Q0 Q1 Q2 Q3 EP ET"
)
symbol_map = {
    π0f:Π0, π1f:Π1, πb0f:ΠB0, πb1f:ΠB1,
    ω0f:Ω0, ω1f:Ω1, ωb0f:ΩB0, ωb1f:ΩB1,
    q0f:Q0, q1f:Q1, q2f:Q2, q3f:Q3,
    ε_P:EP, ε_T:ET,
}
plain_to_coord_map = {
    π0: π0f, π1: π1f, πb0: πb0f, πb1: πb1f,
    ω0: ω0f, ω1: ω1f, ωb0: ωb0f, ωb1: ωb1f,
    q0: q0f, q1: q1f, q2: q2f, q3: q3f,
}
inv_symbol_map = {v: k for k, v in symbol_map.items()}
coord_to_plain_map = {v: k for k, v in plain_to_coord_map.items()}

inner_prod   = ( sp.Matrix([Π0, Π1]).T * ε * sp.Matrix([Ω0, Ω1]) )[0]
inner_prod_B = ( sp.Matrix([ΠB0, ΠB1]).T * ε * sp.Matrix([ΩB0, ΩB1]) )[0]

G = sp.groebner(
    [inner_prod - EP*ET, inner_prod_B - EP*ET, EP**2 - 1, ET**2 - 1],
    Π0,Π1,ΠB0,ΠB1, Ω0,Ω1,ΩB0,ΩB1, EP,ET,
    order="lex",
    domain=sp.EX,
)

@lru_cache(maxsize=None)
def reduce_symb_expr(expr):
    num, den = sp.fraction(sp.together(expr))
    num = G.reduce(sp.expand(num))[1]
    den = G.reduce(sp.expand(den))[1]
    return sp.together(num/den)

def simplify_onshell(expr):
    # Accept both plain-symbol expressions and diffgeom coord-function expressions.
    plain_symbols = set(plain_to_coord_map.keys())
    uses_plain_symbols = bool(expr.free_symbols.intersection(plain_symbols))

    if uses_plain_symbols:
        expr = expr.xreplace(plain_to_coord_map)
        expr = sp.expand(expr.rewrite(sp.exp))

    expr = reduce_symb_expr(expr.subs(symbol_map)).subs(inv_symbol_map)

    if uses_plain_symbols:
        expr = sp.powsimp(sp.trigsimp(sp.simplify(expr), method="fu"), force=True)
        expr = expr.xreplace(coord_to_plain_map)

    return sp.simplify(expr)

def simplify_vf(V):
    V = sp.expand(V)
    V_simp = sum([ simplify_onshell(V.coeff(bv))* bv for bv in C.base_vectors() ])
    return V_simp

def simplify_form(F):
    F = sp.expand(F)
    F_simp = sum([ simplify_onshell(F.coeff(bf))* bf for bf in C.base_oneforms() ])
    print("complete!")
    return F_simp

# -------------------------------------------------
# Caching vector field evaluations for speed up
# -------------------------------------------------

@lru_cache(maxsize=None)
def decompose(V):
    V = sp.expand(V)
    return tuple([(bv, V.coeff(bv)) for bv in C.base_vectors()])

@lru_cache(maxsize=None)
def differentiate(bv, expr):
    return bv.rcall(expr)

@lru_cache(maxsize=None)
def apply_vf(V, expr):
    out = sum((coef * differentiate(bv, expr) for bv, coef in decompose(V) if coef!=0))
    return sp.expand(out)

# -----------------------------
# Build λ and Λ from π and ω
# -----------------------------

# Setting first col of λ to π if ε_Pε_T=+1 or to ω if ε_Pε_T=-1
Πp = (1 + ε_P*ε_T)/2
Πm = (1 - ε_P*ε_T)/2
col1 = Πp*π + Πm*ω; col1b = Πp*πb + Πm*ωb
col2 = Πp*ω + Πm*π; col2b = Πp*ωb + Πm*πb

λ     = sp.Matrix([[col1[0], col2[0]],
                   [col1[1], col2[1]]])

λ_dag = sp.Matrix([[col1b[0], col1b[1]],
                   [col2b[0], col2b[1]]])

def Lambda_col(nu):
    # η is applied to raise the spacetime index on σbar
    col = η * sp.Matrix([sp.trace(σbar[mu] * λ * σ[nu] * λ_dag) / 2 for mu in range(4)])
    # Right multiplication by discrete element corresponding to (ε_P, ε_T)
    if nu == 0:       col = ε_T * col
    if nu in (1,2,3): col = ε_P * col
    return col

def construct_Lambda_cols():
    Λ0 = Lambda_col(0).applyfunc(simplify_onshell)
    Λ1 = Lambda_col(1).applyfunc(simplify_onshell)
    Λ2 = Lambda_col(2).applyfunc(simplify_onshell)
    Λ3 = Lambda_col(3).applyfunc(simplify_onshell)

    compute_time_elapsed("Λ cols constructed")
    return Λ0, Λ1, Λ2, Λ3

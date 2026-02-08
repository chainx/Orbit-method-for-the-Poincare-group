import time
from functools import lru_cache

import sympy as sp
from sympy.diffgeom import Manifold, Patch, CoordSystem

# ---------------------------
# Performance benchmarking
# --------------------------

global total_time_elapsed
total_time_elapsed = 0
start = time.perf_counter()

def time_conversion(s):
    s = int(s)
    hour, r = divmod(s, 3600)
    min, sec = divmod(r, 60)
    out  = f"{hour:d}h " if hour else ""
    out += f"{min}m " if hour or min else ""
    out += f"{sec}s"
    return out

def compute_time_elapsed(msg="", verbose=True):
    global total_time_elapsed
    total_elapsed_new = time.perf_counter() - start
    elapsed = time_conversion(total_elapsed_new-total_time_elapsed)
    if verbose:
        print(f"{msg}. Time elapsed: {elapsed}")
    total_time_elapsed =  total_elapsed_new

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

p0, p1, p2, p3 = sp.symbols("p0 p1 p2 p3", real=True)
q0, q1, q2, q3 = sp.symbols("q0 q1 q2 q3", real=True)
z0, z1, zb0, zb1 = sp.symbols("z0 z1 zb0 zb1")
m = sp.symbols("m", positive=True, real=True)
s = sp.symbols("s", positive=True, real=True)

ε_P = sp.symbols("ε_P", real=True)  # parity label, ε_P^2=1
ε_T = sp.symbols("ε_T", real=True)  # time label, ε_T^2=1

# --------------------------------------------------------------------------
# Diffgeom coordinates (we keep ε_P,ε_T as parameters, not coordinates)
# --------------------------------------------------------------------------

coords = [p0,p1,p2,p3, q0,q1,q2,q3, z0,z1,zb0,zb1]
coord_names = [coord.name for coord in coords]

M = Manifold("M", len(coords))
P = Patch("P", M)
C = CoordSystem("C", P, coords)

p0f,p1f,p2f,p3f, q0f,q1f,q2f,q3f, z0f,z1f,zb0f,zb1f = C.coord_functions()
e_p0,e_p1,e_p2,e_p3, e_q0,e_q1,e_q2,e_q3, e_z0,e_z1,e_zb0,e_zb1 = C.base_vectors()
dp0, dp1, dp2, dp3, dq0, dq1, dq2, dq3, dz0, dz1, dzb0, dzb1    = C.base_oneforms()

sym2bv = dict(zip(coord_names, C.base_vectors()))
sym2bf = dict(zip(coord_names, C.base_oneforms()))
sym2cf = dict(zip(coord_names, C.coord_functions()))

e_p  = sp.Matrix([e_p0,e_p1,e_p2,e_p3])
e_q  = sp.Matrix([e_q0,e_q1,e_q2,e_q3])
e_z  = sp.Matrix([e_z0,e_z1]); e_zb = sp.Matrix([e_zb0,e_zb1])

dp = sp.Matrix([dp0, dp1, dp2, dp3])
dq = sp.Matrix([dq0, dq1, dq2, dq3])
dz, dzb = sp.Matrix([dz0, dz1]), sp.Matrix([dzb0, dzb1])

p = sp.Matrix([p0f,  p1f, p2f, p3f])
z = sp.Matrix([z0f,  z1f]); zb = sp.Matrix([zb0f, zb1f])

# ----------------------------------------
# Groebner ideal for “mod constraints”
# ----------------------------------------

P0,P1,P2,P3,Q0,Q1,Q2,Q3,Z0,Z1,ZB0,ZB1,EP,ET = sp.symbols(
    "P0 P1 P2 P3 Q0 Q1 Q2 Q3 Z0 Z1 ZB0 ZB1 EP ET"
)
symbol_map = {
    p0f:P0, p1f:P1, p2f:P2, p3f:P3,
    q0f:Q0, q1f:Q1, q2f:Q2, q3f:Q3,
    z0f:Z0, z1f:Z1, zb0f:ZB0, zb1f:ZB1,
    ε_P:EP, ε_T:ET,
}
plain_to_coord_map = {
    p0: p0f, p1: p1f, p2: p2f, p3: p3f,
    q0: q0f, q1: q1f, q2: q2f, q3: q3f,
    z0: z0f, z1: z1f, zb0: zb0f, zb1: zb1f,
}
inv_symbol_map = {v: k for k, v in symbol_map.items()}
coord_to_plain_map = {v: k for k, v in plain_to_coord_map.items()}

mass_shell = P0**2 - P1**2 - P2**2 - P3**2 - m**2
pzzb = ET*(sp.Matrix([Z0, Z1]).T * ε.T * sigma(P0, P1, P2, P3) * ε * sp.Matrix([ZB0, ZB1]))[0]
z_constraint = pzzb - s*m/sqrt2

G = sp.groebner(
    [mass_shell, z_constraint, EP**2 - 1, ET**2 - 1],
    P0,P1,P2,P3, Z0,Z1,ZB0,ZB1, EP,ET,
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

    return expr

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

# ------------------------------------------------------------
# w^A(p,z,ε_P) = (sqrt(2)/m) ε_P ε_T p^{A dotA} zb_{dotA}
# ------------------------------------------------------------

w  = (sqrt2/m) * (ε_P*ε_T) * (sigma(p0f, p1f, p2f, p3f)   * ε * zb)
wb = (sqrt2/m) * (ε_P*ε_T) * (sigma(p0f, p1f, p2f, p3f).T * ε * z)

# -----------------------------
# Build λ and Λ from p and z
# -----------------------------

# Setting first col of λ to z if ε_P=+1 or to w if ε_P=-1
Πp = (1 + ε_P)/2
Πm = (1 - ε_P)/2
col1 = Πp*z + Πm*w; col1b = Πp*zb + Πm*wb
col2 = Πp*w + Πm*z; col2b = Πp*wb + Πm*zb

λ     = sp.Matrix([[col1[0], col2[0]],
                   [col1[1], col2[1]]])

λ_dag = sp.Matrix([[col1b[0], col1b[1]],
                   [col2b[0], col2b[1]]])

def Lambda_col(nu):
    # η is applied to raise the spacetime index on σbar
    col = η * sp.Matrix([sp.trace(σbar[mu] * λ * σ[nu] * λ_dag) / 2 for mu in range(4)])
    # Right multiplication by discrete element corresponding to ε_P, ε_T
    if nu == 0:       col = ε_T * col
    if nu in (1,2,3): col = ε_P * col
    return (1/s) * col

def construct_Lambda_cols():
    Λ0 = (1/m) * sp.Matrix([p0f, p1f, p2f, p3f])
    Λ1 = Lambda_col(1).applyfunc(simplify_onshell)
    Λ2 = Lambda_col(2).applyfunc(simplify_onshell)
    Λ3 = Lambda_col(3).applyfunc(simplify_onshell)

    compute_time_elapsed("Λ cols constructed")
    return Λ0, Λ1, Λ2, Λ3

if __name__ == "__main__":
    diff = ( Lambda_col(0) - (1/m) * sp.Matrix([p0f, p1f, p2f, p3f]) ).applyfunc(simplify_onshell)
    assert diff==sp.Matrix([0,0,0,0]), diff

from performance_benchmarking import compute_time_elapsed
from poincare_algebra import J, T0, T

from itertools import product
import sympy as sp

I = sp.I
half = sp.S.Half

# --------------------
# Define symbols
# --------------------

m = sp.symbols("m", real=True)
s = sp.symbols("s", positive=True, real=True)
ε_P = sp.symbols("ε_P", real=True)
p0 = sp.symbols("p0", real=True, nonzero=True)
p1, p2, p3 = sp.symbols("p1 p2 p3", real=True)
p = sp.Matrix([p0, p1, p2, p3])
p⃗ = sp.Matrix([p1, p2, p3])

# --------------------------------------------
# Define mapping between p^μ and p^{A\dotA}
# --------------------------------------------

def sigma(p):
    return (1/sp.sqrt(2)) * sp.Matrix([
        [p[0] + p[3],   p[1] - I*p[2]],
        [p[1] + I*p[2], p[0] - p[3]],
    ])

# ------------------------------------------
# Impose on-shell constraints and simplify
# ------------------------------------------

# Symbols to replace non-polynomial terms
ε_T = sp.symbols("ε_T", real=True) # ε_T := sgn(p^0)
sqrt_term_symbol = sp.symbols("sqrt_term_symbol")
sqrt_term = sp.sqrt(m + p0*ε_T)
G = sp.groebner(
    [
        p0**2 - p1**2 - p2**2 - p3**2 - m**2,
        ε_T**2 - 1, ε_P**2 - 1,
        sqrt_term_symbol**2 - sqrt_term**2,
    ],
    p0, p1, p2, p3, ε_T, ε_P, sqrt_term_symbol,
    order="lex",
    domain=sp.EX,
)
def reduce_onshell(expr):
    expr = sp.together(expr).subs({sp.Abs(p0): p0*ε_T, sp.sign(p0): ε_T})
    expr = expr.subs({sqrt_term: sqrt_term_symbol}) 
    expr = expr.xreplace({sp.conjugate(s): s for s in (p0,p1,p2,p3,sqrt_term_symbol,m,ε_T)})

    num, den = sp.fraction(sp.together(expr))
    num = G.reduce(sp.expand(num))[1]
    den = G.reduce(sp.expand(den))[1]

    out = sp.together(num/den).subs({ε_T**2: 1, ε_T**-1: ε_T, sqrt_term_symbol: sqrt_term})
    return sp.together(out)

def simplify(expr):
    expr = sp.together(expr).subs({sp.conjugate(sp.sqrt(1/m)): sp.sqrt(1/m)})
    expr = sp.cancel(expr.rewrite(sp.exp))
    expr = sp.powsimp(expr, force=True)
    return sp.simplify(sp.together(expr))

# -------------------------------------------------------------------------
# Verification of Pauli-Lubanski vectors for f_(m,s), f_(+,s) and f_(-,s)
# -------------------------------------------------------------------------

η = sp.diag(1,-1,-1,-1) # Minkowski metric

def pauli_lubanski_vector(p, M):
    """ W := -*(p ∧ M) """
    W = sp.zeros(4, 1)
    for mu, nu, rho, sigma in product(range(4), repeat=4):
        # Using η to convert p^μ and M^μ_ν to p_μ and M_{μν}
        W[mu] += -half*sp.LeviCivita(mu,nu,rho,sigma)*(η*p)[nu]*(η*M)[rho, sigma]
    return W

p_m, M_s = m*T0, s*J[2]
assert pauli_lubanski_vector(p_m, M_s) == sp.Matrix([0, 0, 0, m*s])
compute_time_elapsed("Verified Pauli-Lubanksi vector for f_(m,s)")

p_N = T0 + T[2]
assert pauli_lubanski_vector(p_N, M_s) == s * p_N
compute_time_elapsed("Verified Pauli-Lubanksi vector for f_(+,s)")
p_N = -T0 + T[2]
assert pauli_lubanski_vector(p_N, M_s) == -s * p_N
compute_time_elapsed("Verified Pauli-Lubanksi vector for f_(-,s)")

# -------------------------------------------------------------------------------------
# Define the 4-vector frak{p}^μ corresponding to the square root of sgn(p^0)*sigma(p)
# -------------------------------------------------------------------------------------

frakp0 = 2**(-sp.Rational(1, 4)) * sp.sqrt(sp.Abs(p0) + m)
frakp1 = 2**(-sp.Rational(1, 4)) * sp.sign(p0) * p1 / sp.sqrt(sp.Abs(p0) + m)
frakp2 = 2**(-sp.Rational(1, 4)) * sp.sign(p0) * p2 / sp.sqrt(sp.Abs(p0) + m)
frakp3 = 2**(-sp.Rational(1, 4)) * sp.sign(p0) * p3 / sp.sqrt(sp.Abs(p0) + m)
frakp = sp.Matrix([frakp0, frakp1, frakp2, frakp3])

sqrt_p = sigma(frakp)

# ----------------------------------------------
# Verification of sqrt_p^2 = sgn(p^0)*sigma(p)
# ----------------------------------------------

diff = (sqrt_p*sqrt_p - sp.sign(p0) * sigma(p)).applyfunc(reduce_onshell)
assert diff == sp.zeros(2), diff
compute_time_elapsed("Verified (sqrt_p)^2 == sgn(p0)σ(p)")

# --------------------------------------------------------
# Verification of formula for Pauli-Lubanski vector W^μ
# --------------------------------------------------------

s = sp.symbols("s", positive=True, nonzero=True)
θ, χ0, χ1 = sp.symbols("θ χ0 χ1", real=True)
ϕ = χ1 - χ0

u = sp.sqrt(sp.sqrt(2)*s/m) * sp.Matrix([
    sp.exp(I*χ0)*sp.cos(θ/2),
    sp.exp(I*χ1)*sp.sin(θ/2),
])
z = sqrt_p * u

s⃗ = s * sp.Matrix([
    sp.sin(θ) * sp.cos(ϕ),
    sp.sin(θ) * sp.sin(ϕ),
    sp.cos(θ),
])
u_vec = (1/m) * sp.Matrix([s, s⃗[0], s⃗[1], s⃗[2]])
diff = (u*u.H - sigma(u_vec)).applyfunc(simplify)
assert diff == sp.zeros(2), diff
compute_time_elapsed("Verified s⃗ is the 3-vector associated to u")

W0 = ε_P * p⃗.dot(s⃗)
W⃗ = ε_P * sp.sign(p0) * ( m*s⃗ + (p⃗.dot(s⃗)/(sp.Abs(p[0]) + m))*p⃗ )
W = sp.Matrix([W0, W⃗[0], W⃗[1], W⃗[2]])

n = (s/(sp.sqrt(2)*m)) * (p + ε_P * W/s) # Null vector defined in equation 2.3.2

diff = (sp.sign(p0) * z*z.H - sigma(n)).applyfunc(reduce_onshell).applyfunc(simplify)
assert diff == sp.zeros(2), diff
compute_time_elapsed("Verified sgn(p^0) z^A zbar^\dotA == σ(n)")
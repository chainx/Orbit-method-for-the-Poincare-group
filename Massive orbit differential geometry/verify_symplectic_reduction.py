from performance_benchmarking import compute_time_elapsed

from massive_orbits_dg import (
    C, coords, coord_to_plain_map, decompose,
    sigma, sp, I, sqrt2, η, ε,
    p0, p1, p2, p3, z0, z1, zb0, zb1,
    m, ε_P, ε_T,
)
from verify_vector_fields_and_forms import Θ_T0, Θ_J3, V_T0, V_J3, s

# Extend the actual potential from the table by m -> sqrt(p_μ p^μ).
# The spin parameter cancels from m Θ_T0 + s Θ_J3 before extension.
Θ_ms = sp.expand(m * Θ_T0 + s * Θ_J3)
m_p = sp.sqrt(p0**2 - p1**2 - p2**2 - p3**2)
# SymPy 1.9 can miss one-form coefficients in a sum. Extract them termwise.
potential_terms = sp.Add.make_args(Θ_ms)
θ = sp.Matrix([
    sum(term.coeff(bf) for term in potential_terms)
    .xreplace(coord_to_plain_map).subs(m, m_p)
    for bf in C.base_oneforms()
])
assert all(s not in coefficient.free_symbols for coefficient in θ)

# Ω_ij = -∂_i θ_j + ∂_j θ_i. Contracting the first slot gives Ω.T * V.
Ω = θ.jacobian(coords) - θ.jacobian(coords).T


def simplify_component(expr):
    # ε_P and ε_T label components, so their squares are one.
    expr = sp.cancel(sp.simplify(expr))
    return sp.simplify(expr.subs({ε_P**4: 1, ε_T**4: 1, ε_P**2: 1, ε_T**2: 1}))


def coefficients(V):
    return sp.Matrix([
        coefficient.xreplace(coord_to_plain_map).subs(m, m_p)
        for _, coefficient in decompose(V)
    ])


p = sp.Matrix([p0, p1, p2, p3])
z = sp.Matrix([z0, z1])
zb = sp.Matrix([zb0, zb1])
C_m = m_p
C_s = (sqrt2/m_p) * ε_T * (z.T * ε.T * sigma(*p) * ε * zb)[0]
V_m = coefficients(V_T0)
V_s = coefficients(V_J3)

for name, V, constraint in (("mass", V_m, C_m), ("spin", V_s, C_s)):
    gradient = sp.Matrix([sp.diff(constraint, x) for x in coords])
    contraction = (θ.T * V)[0]
    assert simplify_component(contraction - constraint) == 0, name
    defect = (Ω.T * V - gradient).applyfunc(simplify_component)
    assert defect == sp.zeros(12, 1), (name, defect)
compute_time_elapsed("i_V Ω = dC for mass and spin constraints verified")

# With i_X Ω = -dC, X_m = -V_m. Thus {C_m,C_s} = -V_m(C_s).
poisson_bracket = -(V_m.T * sp.Matrix([sp.diff(C_s, x) for x in coords]))[0]
assert simplify_component(poisson_bracket) == 0
compute_time_elapsed("Mass and spin constraints Poisson commute")

# The canonical p,q block and the absence of q,spin cross terms imply that
# Ω is non-degenerate precisely when its spinor block is non-degenerate.
assert Ω[:4, 4:8] == -η
assert Ω[4:8, :4] == η
assert Ω[4:8, 4:] == sp.zeros(4, 8)
spin_block = Ω[8:, 8:]
assert simplify_component(spin_block.det()) == 16
compute_time_elapsed("Extended symplectic form is non-degenerate for timelike p")

# At z = 0 the mixed blocks vanish. The spinor block stays invertible, so
# the four real spinor constraints are second class; their restriction
# leaves only the canonical p,q form.
zero_spinor = {z0: 0, z1: 0, zb0: 0, zb1: 0}
Ω_at_zero = Ω.subs(zero_spinor).applyfunc(simplify_component)
assert Ω_at_zero[:8, 8:] == sp.zeros(8, 4)
assert Ω_at_zero[8:, :8] == sp.zeros(4, 8)
canonical_block = sp.zeros(8)
canonical_block[:4, 4:] = -η
canonical_block[4:, :4] = η
assert Ω_at_zero[:8, :8] == canonical_block
assert simplify_component(Ω_at_zero[8:, 8:].det()) == 16
assert simplify_component(C_s.subs(zero_spinor)) == 0
compute_time_elapsed("Spinless second-class reduction leaves -dp_μ wedge dq^μ")

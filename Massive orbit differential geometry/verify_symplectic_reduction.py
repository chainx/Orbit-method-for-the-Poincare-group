from performance_benchmarking import compute_time_elapsed

from massive_orbits_dg import (
    dot, sigma, simplify_onshell,
    sp, I, sqrt2, η, ε,
    p0, p1, p2, p3, q0, q1, q2, q3,
    m, s, ε_P, ε_T,
)
from verify_vector_fields_and_forms import Θ_T0, Θ_J3, V_J3

from sympy.diffgeom import Differential

dΘ_ms = Differential(m * Θ_T0 + s * Θ_J3)

Ω = 1
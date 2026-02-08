import time
from functools import lru_cache

from massive_orbits_dg import (
    dot, sigma, simplify_onshell, compute_time_elapsed,
    sp, I, sqrt2, η, ε,
    p0, p1, p2, p3, q0, q1, q2, q3,
    m, s, ε_P, ε_T,
)

θ, χ0, χ1 = sp.symbols("θ χ0 χ1", real=True)
coords = [p0, p1, p2, p3, q0, q1, q2, q3, θ, χ0, χ1]

# --------------------------------------------------------------
# z(p,θ,χ0,χ1) and w(p,z,ε_P,ε_T) with sqrt_p^2 = ε_T sigma(p)
# --------------------------------------------------------------

frakp0 = 2 ** (-sp.Rational(1, 4)) * sp.sqrt(ε_T * p0 + m)
frakp1 = 2 ** (-sp.Rational(1, 4)) * ε_T * p1 / sp.sqrt(ε_T * p0 + m)
frakp2 = 2 ** (-sp.Rational(1, 4)) * ε_T * p2 / sp.sqrt(ε_T * p0 + m)
frakp3 = 2 ** (-sp.Rational(1, 4)) * ε_T * p3 / sp.sqrt(ε_T * p0 + m)
sqrt_p = sigma(frakp0, frakp1, frakp2, frakp3)

u = sp.sqrt(sqrt2 * s / m) * sp.Matrix([
    sp.exp(I * χ0) * sp.cos(θ / 2),
    sp.exp(I * χ1) * sp.sin(θ / 2),
])
ub = sp.sqrt(sqrt2 * s / m) * sp.Matrix([
    sp.exp(-I * χ0) * sp.cos(θ / 2),
    sp.exp(-I * χ1) * sp.sin(θ / 2),
])

# z = sqrt_p u,  zbar = (sqrt_p)bar ubar = sqrt_p.T ubar
z = sqrt_p * u
zb = sqrt_p.T * ub

pσ = sigma(p0, p1, p2, p3)
w  = (sqrt2 / m) * (ε_P * ε_T) * (pσ * ε * zb)
wb = (sqrt2 / m) * (ε_P * ε_T) * (pσ.T * ε * z)

# --------------------------------------------------------------------
# Θ_ms = m Θ_T0 + s Θ_J3 = p_μ dq^μ + i(w_A dz^A - c.c) (eq. 2.3.31)
# --------------------------------------------------------------------

dz  = z.jacobian(coords)
dzb = zb.jacobian(coords)
spin_coeff = I * (w.T * ε * dz - wb.T * ε * dzb)

# p_μ dq^μ
p_dq = sp.Matrix([0,0,0,0,p0,-p1,-p2,-p3,0,0,0])

Θ_ms_coeff = p_dq + spin_coeff.T

# -------------------------------------------------------------------
# pre_symplectic_form in (p,q,θ,χ0,χ1), using ϕ=χ1-χ0, ψ=χ1+χ0
# eq. (2.3.33): p_μ dq^μ + s ε_P (cosθ dφ - dψ - ω·ŝ)
# -------------------------------------------------------------------

ϕ = χ1 - χ0
ŝ = sp.Matrix([
    sp.sin(θ) * sp.cos(ϕ),
    sp.sin(θ) * sp.sin(ϕ),
    sp.cos(θ),
])

# p_μ dq^μ contribution
pre_symplectic_coeff = sp.Matrix([0,0,0,0,p0,-p1,-p2,-p3,0,0,0])

# (dp x p) . ŝ = (p x ŝ) . dp contribution
p_vec = sp.Matrix([p1, p2, p3])
ω_factor = -(s * ε_P) / (m * (ε_T * p0 + m))
for i in range(3):
    pre_symplectic_coeff[i+1] += ω_factor * p_vec.cross(ŝ)[i]

# cosθ dφ - dψ = (-1-cosθ)dχ0 + (cosθ-1)dχ1 contribution
pre_symplectic_coeff[9]  += s * ε_P * (-1 - sp.cos(θ))
pre_symplectic_coeff[10] += s * ε_P * (sp.cos(θ) - 1)

# ---------------
# Verification
# ---------------

diff_coeff = (Θ_ms_coeff - pre_symplectic_coeff).applyfunc(simplify_onshell)
assert diff_coeff == sp.zeros(11,1)
compute_time_elapsed("Pre-symplectic form verified")
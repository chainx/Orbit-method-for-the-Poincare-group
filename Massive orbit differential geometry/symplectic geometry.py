from sympy.diffgeom import Differential, WedgeProduct
from sympy import cos, sin, exp, sqrt

from massive_orbits_dg import *
from verify_vector_fields_and_forms import VFs, Θ_T0, Θ_J3

def cross3(a,b):
    ax,ay,az = a[1:]; bx,by,bz = b[1:]
    return sp.Matrix([ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx])

def sigma_inv(σv):
    return η * sp.Matrix([ sp.Trace(σv*σbar[mu]) for mu in range(4) ])

# ----------------------------------
# Constructing pre-symplectic form
# ----------------------------------

frakp0 = 2**(-sp.Rational(1, 4)) * sp.sqrt(ε_T*p0f + m)
frakp1 = 2**(-sp.Rational(1, 4)) * ε_T * p1f / sp.sqrt(ε_T*p0f + m)
frakp2 = 2**(-sp.Rational(1, 4)) * ε_T * p2f / sp.sqrt(ε_T*p0f + m)
frakp3 = 2**(-sp.Rational(1, 4)) * ε_T * p3f / sp.sqrt(ε_T*p0f + m)
sqrt_p = sigma(frakp0, frakp1, frakp2, frakp3)

# z = sqrt_p * u
u = sqrt_p.LUsolve(z)

"""
u = sqrt(√2s/m) ( e^iχ0 cos(θ/2), e^iχ1 sin(θ/2) )
ŝ = ( sinθ*cosϕ, sinθ*sinϕ, cosθ )
"""
ŝ = (m/s) * sigma_inv(u*u.H)[1:, :]
ŝ = ŝ.applyfunc(simplify_onshell)
cosθ = ŝ[2]
sinθ = simplify_onshell(sqrt(1-cosθ**2))
compute_time_elapsed("cosθ and sinθ constructed")

# We have to assume θ is not 0 or pi
exp_iψ = u[0]*u[1]/(2*sinθ)
exp_iψ = simplify_onshell(exp_iψ)
dψ = (-I/exp_iψ) * Differential(exp_iψ)
dψ = simplify_form(dψ)
compute_time_elapsed("dψ constructed")


cosϕ, sinϕ = ŝ[0]/sinθ, ŝ[2]/sinθ
cosϕ, sinϕ = simplify_onshell(cosϕ), simplify_onshell(sinϕ)
dϕ = (1/cosϕ) * Differential(sinϕ)
dϕ = simplify_form(dϕ)
compute_time_elapsed("dϕ constructed")

ω  = (1/(m*(p0 + m))) * cross3(dp, p)

pre_symplectic_form = dot(η*p, dq) + (s*ε_P)*( cosθ*dϕ - dψ - dot(ω, ŝ))
compute_time_elapsed("pre-symplectic form constructed")

pre_symplectic_form = simplify_form(pre_symplectic_form)

# ----------------------------------
# Verifying pre-symplectic form
# ----------------------------------

Θ_ms = m * Θ_T0 + s * Θ_J3

diff = Θ_ms - pre_symplectic_form
for VF_name, VF in VFs.items():
    print(VF_name)
    assert diff.rcall(VF)==0, VF_name

# -------------------------
# Computing Hamiltonians
# -------------------------

ω = Differential(Θ_ms)
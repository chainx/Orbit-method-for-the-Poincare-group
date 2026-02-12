from performance_benchmarking import compute_time_elapsed

import sympy as sp
from sympy import euler_equations

tau = sp.symbols('tau', real=True)
m = sp.symbols('m', positive=True, nonzero=True)

q0 = sp.Function('q0')(tau)
q̇1,q̇2,q̇3 = (sp.Function(n)(tau) for n in ('q̇1','q̇2','q̇3'))
p1,p2,p3 = (sp.Function(n)(tau) for n in ('p1','p2','p3'))
s1,s2,s3 = (sp.Function(n)(tau) for n in ('s1','s2','s3'))
p = sp.Matrix([p1,p2,p3]); ṗ = sp.diff(p, tau)
s = sp.Matrix([s1,s2,s3]); ṡ = sp.diff(s, tau)
q̇ = sp.Matrix([sp.diff(q̇1,tau), sp.diff(q̇2,tau), sp.diff(q̇3,tau)]) ; q̇0 = sp.diff(q0, tau)

p0 = sp.sqrt(p.dot(p) + m**2)
ω = ṗ.cross(p) / (m*(p0+m)) # Thomas precession angular frequency vector
ṡ_constraint = ω.cross(s) # Obtained from EL equations for Θ and ϕ in full action

Lagrangian = p0*q̇0 - p.dot(q̇) - ω.dot(s)
EL = euler_equations(Lagrangian, (p1,p2,p3), tau)
EL = [sp.simplify(eq.lhs.xreplace({ṡ[n]: ṡ_constraint[n] for n in range(3)})) for eq in EL]

# Convert EL equation into the form Aq̇ = b, then solve for q̇
A, b = sp.linear_eq_to_matrix(EL, [sp.diff(q̇1,tau), sp.diff(q̇2,tau), sp.diff(q̇3,tau)])
q̇_solved = A.LUsolve(b)            # Aq̇ = b => q̇ = A^{-1} b
assert sp.simplify(A*q̇_solved - b)==sp.Matrix([[0], [0], [0]]), (A*q̇_solved, b)

# Verification of q̇ - (q̇0/p0)p = ( [p.s]ω + ṗxs ) / (m*p0) after using EL equations to solve for q̇_solved
LHS = q̇_solved - (q̇0/p0)*p
RHS = ( p.dot(s)*ω + ṗ.cross(s) ) / (m*p0)
assert sp.simplify(LHS - RHS)==sp.Matrix([[0], [0], [0]]), (LHS, RHS)

compute_time_elapsed("Momentum EL equation implies\nq̇ - (q̇0/p0)p = ( [p.s]ω + ṗxs ) / (m*p0)\nverified")
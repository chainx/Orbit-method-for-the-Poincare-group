from performance_benchmarking import compute_time_elapsed

import sympy as sp
from sympy import euler_equations

tau = sp.symbols('tau', real=True)
m = sp.symbols('m', positive=True, nonzero=True)
ε_P, ε_T = sp.symbols('ε_P ε_T', real=True)

q0 = sp.Function('q0')(tau)
q̇1,q̇2,q̇3 = (sp.Function(n)(tau) for n in ('q̇1','q̇2','q̇3'))
p1,p2,p3 = (sp.Function(n)(tau) for n in ('p1','p2','p3'))
s1,s2,s3 = (sp.Function(n)(tau) for n in ('s1','s2','s3'))
p = sp.Matrix([p1,p2,p3]); ṗ = sp.diff(p, tau)
s = sp.Matrix([s1,s2,s3]); ṡ = sp.diff(s, tau)
q̇ = sp.Matrix([sp.diff(q̇1,tau), sp.diff(q̇2,tau), sp.diff(q̇3,tau)]) ; q̇0 = sp.diff(q0, tau)

# The component labels are constant along the path. E=|p^0| on either sheet.
E = sp.sqrt(p.dot(p) + m**2)
p0 = ε_T * E
ω = ṗ.cross(p) / (m*(E+m)) # Thomas precession angular frequency vector
ṡ_constraint = ω.cross(s) # Obtained from EL equations for Θ and ϕ in full action

Lagrangian = p0*q̇0 - p.dot(q̇) - ε_P*ω.dot(s)
EL = euler_equations(Lagrangian, (p1,p2,p3), tau)
EL = [sp.simplify(eq.lhs.xreplace({ṡ[n]: ṡ_constraint[n] for n in range(3)})) for eq in EL]

# Older SymPy versions require symbols as the matrix unknowns.
# Substitute only after taking the Euler–Lagrange derivatives.
velocity_symbols = [sp.Dummy(f'v{i}') for i in range(1, 4)]
velocity_substitutions = dict(zip(q̇, velocity_symbols))
linear_EL = [eq.xreplace(velocity_substitutions) for eq in EL]
A, b = sp.linear_eq_to_matrix(linear_EL, velocity_symbols)
q̇_solved = A.LUsolve(b)            # Aq̇ = b => q̇ = A^{-1} b
assert sp.simplify(A*q̇_solved - b)==sp.Matrix([[0], [0], [0]]), (A*q̇_solved, b)

# Since ε_T^2=1, 1/p0=ε_T/E. Keep both component signs throughout the
# variation: the spin term carries ε_P and its denominator uses |p^0|.
LHS = q̇_solved - (ε_T*q̇0/E)*p
RHS = ε_P * ( p.dot(s)*ω + ṗ.cross(s) ) / (m*E)
assert sp.simplify(LHS - RHS)==sp.Matrix([[0], [0], [0]]), (LHS, RHS)

compute_time_elapsed("Momentum EL equation implies\nq̇ - (q̇0/p0)p = ε_P ( [p.s]ω + ṗxs ) / (m*|p0|)\nverified for both parity and energy signs")

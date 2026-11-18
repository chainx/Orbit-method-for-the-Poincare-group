from performance_benchmarking import compute_time_elapsed

import sympy as sp
I = sp.I
τ = sp.symbols("τ")

def Lorentz_LeviCivita(mu,nu,rho,sig):
    """ ε^{0123} = +1 => ε_{0123} = -1. """
    return -sp.LeviCivita(mu,nu,rho,sig)

# ----------------------------
# Symbols: p, ṗ, q̇, s, ṡ, m
# ----------------------------

m = sp.symbols("m", nonzero=True) # Mass
s = sp.symbols("s", nonzero=True) # Spin
ε_P, ε_T = sp.symbols("ε_P ε_T", real=True)


p0, p1, p2, p3 = (sp.Function(f"p{μ}")(τ) for μ in range(4))
ṗ0, ṗ1, ṗ2, ṗ3 = (sp.diff(pμ, τ) for pμ in (p0, p1, p2, p3))
p = sp.Matrix([p1, p2, p3])
ṗ = sp.Matrix([ṗ1, ṗ2, ṗ3])
p_contra = sp.Matrix([p0, p1, p2, p3])
ṗ_contra = sp.Matrix([ṗ0, ṗ1, ṗ2, ṗ3])
p_cov    = sp.Matrix([p0, -p1, -p2, -p3])

q0, q1, q2, q3 = (sp.Function(f"q{μ}")(τ) for μ in range(4))
q̇0, q̇1, q̇2, q̇3 = (sp.diff(qμ, τ) for qμ in (q0, q1, q2, q3))
q̇     = sp.Matrix([q̇1, q̇2, q̇3])
q̇_cov = sp.Matrix([q̇0, -q̇1, -q̇2, -q̇3])

s1, s2, s3 = (sp.Function(f"s{i}")(τ) for i in range(3))
ṡ1, ṡ2, ṡ3 = (sp.diff(si, τ) for si in (s1, s2, s3))
s = sp.Matrix([s1, s2, s3])
ṡ = sp.Matrix([ṡ1, ṡ2, ṡ3])

# -------------------------------------------
# Mass-shell: pμ p^μ = p0^2 - |p|^2 = m^2
# and its derivative: p·ṗ = p0 ṗ0
# -------------------------------------------

mass_shell_rule      = {p.dot(p): p0**2 - m**2}
mass_shell_dot_rule  = {p.dot(ṗ): p0*ṗ0}
ṗ0_rule              = {ṗ0: p.dot(ṗ)/p0}  # solve derivative constraint for ṗ0
mass_shell_poly_rule = {m**2 - p0**2 + p1**2 + p2**2 + p3**2: 0}

# ----------------------------
# Definitions from the paper
# ----------------------------

E = ε_T*p0 # |p^0| on the component with sign(p^0)=ε_T.
ω  = (1/(m*(E + m))) * ṗ.cross(p) # Thomas precession vector

# Formula for W^μ verified in pauli_lubanski_vector.py in all components.
W0 = ε_P*p.dot(s)
W  = ε_P*ε_T*(m*s + (p.dot(s)/(E + m))*p)
W_contra = sp.Matrix([W0, W[0], W[1], W[2]])

# S_{μν} = -(1/m^2) ε_{μνρσ} p^ρ W^σ
def S_mu_nu(mu, nu):
    tot = 0
    for rho in range(4):
        for sig in range(4):
            tot += -Lorentz_LeviCivita(mu,nu,rho,sig) * p_contra[rho] * W_contra[sig]
    return sp.simplify(tot / m**2)

# LHS: dotS_{μν} = d/dτ S_{μν}
def dotS_mu_nu(mu, nu):
    return sp.simplify(sp.diff(S_mu_nu(mu, nu), τ))

# RHS: (q̇ ∧ p)_{μν} = q̇_μ p_ν - q̇_ν p_μ 
def wedge_q̇_p(mu, nu):
    return sp.simplify(q̇_cov[mu]*p_cov[nu] - q̇_cov[nu]*p_cov[mu])

# ----------------------------
# ODEs as substitution rules
# ----------------------------

# Thomas precession: ṡ = ω × s
thomas_rule = {
    ṡ1: ω.cross(s)[0],
    ṡ2: ω.cross(s)[1],
    ṡ3: ω.cross(s)[2],
}

# Hidden momentum: q̇ - (q̇0/p0) p = ε_P/(m |p0|) ( (p·s) ω + ṗ × s )
# Verified in momentum_EL_equations.py
hidden_momentum_rule = {
    q̇1: (q̇0/p0)*p1 + (ε_P/(m*E)) * (p.dot(s)*ω + ṗ.cross(s))[0],
    q̇2: (q̇0/p0)*p2 + (ε_P/(m*E)) * (p.dot(s)*ω + ṗ.cross(s))[1],
    q̇3: (q̇0/p0)*p3 + (ε_P/(m*E)) * (p.dot(s)*ω + ṗ.cross(s))[2],
}

# ---------------------------------------------------------------
# Verification: check μ<ν components vanish after substitutions
# ---------------------------------------------------------------

def simp(expr):
    # Reduce the two component signs before factoring, including powers in
    # rational denominators. This checks every (ε_P,ε_T) simultaneously.
    num, den = sp.fraction(sp.together(expr))
    def reduce_signs(poly):
        return sp.expand(poly).replace(
            lambda term: term.is_Pow and term.base in (ε_P, ε_T)
            and term.exp.is_Integer,
            lambda term: term.base**(term.exp % 2),
        )
    expr = reduce_signs(num)/reduce_signs(den)
    return sp.simplify(sp.together(sp.factor(expr)))

diffs = {} 
for mu in range(4):
    for nu in range(mu+1, 4):
        expr = simp(dotS_mu_nu(mu,nu) - wedge_q̇_p(mu,nu))

        expr = expr.subs(thomas_rule).subs(hidden_momentum_rule)

        expr = expr.subs(mass_shell_dot_rule).subs(mass_shell_rule)
        expr = simp(expr.subs(ṗ0_rule).subs(mass_shell_rule))
        diffs[(mu,nu)] = simp(expr.subs(mass_shell_poly_rule))

assert set(diffs.values()) == {0}, diffs

compute_time_elapsed("EL equations imply Ṡ = q̇ ∧ p verified for both parity and energy signs")

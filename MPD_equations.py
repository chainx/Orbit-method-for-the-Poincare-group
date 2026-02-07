import sympy as sp

I = sp.I
τ = sp.symbols("τ")

# -----------------------------------------------
# Helper functions for 3-vectors and 4-vectors
# -----------------------------------------------

def dot3(a, b):
    return sp.simplify((a.T*b)[0])

def cross3(a,b):
    ax,ay,az = a; bx,by,bz = b
    return sp.Matrix([ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx])

def wedge(a,b):
    # (a ∧ b)_{μν} with contravariant components
    return sp.Matrix(4,4, lambda i,j: sp.simplify(a[i]*b[j] - a[j]*b[i]))

# ----------------------------
# Levi-Civita conventions
# ----------------------------

def Lorentz_LeviCivita(mu,nu,rho,sig):
    """
    Weird convention issue!
    """
    return sp.LeviCivita(mu,nu,rho,sig)

# ----------------------------
# Symbols: p, ṗ, q̇, s, ṡ, m
# ----------------------------

m = sp.symbols("m", nonzero=True) # Mass
s = sp.symbols("s", nonzero=True) # Spin


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

mass_shell_rule      = {dot3(p,p): p0**2 - m**2}
mass_shell_dot_rule  = {dot3(p,ṗ): p0*ṗ0}
ṗ0_rule              = {ṗ0: dot3(p,ṗ)/p0}  # solve derivative constraint for ṗ0
mass_shell_poly_rule = {m**2 - p0**2 + p1**2 + p2**2 + p3**2: 0}

# ----------------------------
# Definitions from the paper
# ----------------------------

ω  = (1/(m*(p0 + m))) * cross3(ṗ, p) # Thomas precession vector

# Formula for W^μ verified in pauli_lubanski_vector.py
W0 = dot3(p, s)
W  = m*s + (W0/(p0 + m))*p
W_contra = sp.Matrix([W0, W[0], W[1], W[2]])

# S_{μν} = (1/m^2) ε_{μνρσ} p^ρ W^σ
def S_mu_nu(mu, nu):
    tot = 0
    for rho in range(4):
        for sig in range(4):
            tot += Lorentz_LeviCivita(mu,nu,rho,sig) * p_contra[rho] * W_contra[sig]
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
    ṡ1: cross3(ω, s)[0],
    ṡ2: cross3(ω, s)[1],
    ṡ3: cross3(ω, s)[2],
}

# Hidden momentum: q̇ - (q̇0/p0) p = (1/(m p0)) ( (p·s) ω + ṗ × s )
hidden_momentum_rule = {
    q̇1: (q̇0/p0)*p1 + (1/(m*p0)) * (dot3(p,s)*ω + cross3(ṗ, s))[0],
    q̇2: (q̇0/p0)*p2 + (1/(m*p0)) * (dot3(p,s)*ω + cross3(ṗ, s))[1],
    q̇3: (q̇0/p0)*p3 + (1/(m*p0)) * (dot3(p,s)*ω + cross3(ṗ, s))[2],
}

# ---------------------------------------------------------------
# Verification: check μ<ν components vanish after substitutions
# ---------------------------------------------------------------

def simp(expr):
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

print("Success!")
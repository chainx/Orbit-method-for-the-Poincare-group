from performance_benchmarking import compute_time_elapsed

import sympy as sp
from functools import reduce, partial

def compose_funcs(f, g):
    return lambda p, z, ε_P: f(*g(p, z, ε_P))
def compose(*funcs):
    return reduce(compose_funcs, funcs)

I = sp.I # Imaginary number

def main():
    verify_sigma_identity()
    verify_relations()

# =====================   DEFINITIONS   =====================

p0 = sp.symbols("p0", real=True, nonzero=True) # p^μ is assumed timelike
p1, p2, p3 = sp.symbols("p1 p2 p3", real=True)
p = sp.Matrix([p0, p1, p2, p3])
z0, z1 = sp.symbols("z0 z1"); z = sp.Matrix([z0, z1])
ψ0, ψ1 = sp.symbols("ψ0 ψ1"); ψ = sp.Matrix([ψ0, ψ1])

ε_P = sp.symbols('ε_P', real=True)
m = sp.symbols("m", real=True)

ε = sp.Matrix([[0, 1],
               [-1, 0]])
R = ε # R carries left and right handed spinor indices, but as a matrix it's the same as ε

def sigma(p):
    """ The Hermitian matrix p^{A\dotA} = p^μ \sigma_μ^{A\dotA} associated to the 4-momentum p """
    return (1/sp.sqrt(2)) * sp.Matrix([
        [p[0] + p[3],   p[1] - I*p[2]],
        [p[1] + I*p[2], p[0] - p[3]],
    ])

def w(p, z, ε_P):
    # p^{A\dotA} \bar{z}_\dotA = p^{A\dotA} (- ε_{\dotA \dotB} \bar{z}^\dotB )
    return ε_P * sp.sign(p[0]) * (sp.sqrt(2) / m) * (sigma(p) * -ε * sp.conjugate(z))

# =====================   PARITY / TIME REVERSAL ACTIONS   =====================

""" a, b, c label the double covers of the Poincare group """

def act_T(p, z, ε_P, a, b, c):
    """ Left action of time reversal """
    Tp = sp.diag(-1, 1, 1, 1) * p
    double_cover_sign = sp.sign(p[0])**b * ε_P**(a+b+c)
    Tz = double_cover_sign * ε_P*R*sp.conjugate(w(p, z, ε_P))
    return Tp, Tz, ε_P

def act_P(p, z, ε_P, a, b, c):
    """ Left action of parity reversal """
    Pp = sp.diag(1, -1, -1, -1) * p
    double_cover_sign = ε_P**a
    Pz = double_cover_sign * -ε_P*R*sp.conjugate(z)
    return Pp, Pz, -ε_P

def act_gauge_transfo(p, z, ε_P, a, b, c):
    """ Right action of parity reversal """
    double_cover_sign = ε_P**a * sp.sign(p[0])**(a+b+c)
    Gz = double_cover_sign * -w(p,z,ε_P)
    return p, Gz, -ε_P

# =====================   ALGEBRAIC SIMPLIFICATION   =====================

ε_T = sp.symbols('ε_T', real=True)
zb0, zb1, ψb0, ψb1 = sp.symbols("zb0 zb1 ψb0 ψb1")
Grob = sp.groebner(
    [p0**2 - p1**2 - p2**2 - p3**2 - m**2, ε_P**2 - 1, ε_T**2 - 1],
    z0, z1, zb0, zb1, p0, p1, p2, p3, ε_P, ε_T, m, ψ0, ψ1, ψb0, ψb1,
    order="lex",
    domain=sp.QQ.algebraic_field(I),
)
def impose_constraints(expr):
    expr = sp.together(sp.simplify(expr))
    subs = {
        sp.Abs(p0): p0*ε_T,
        sp.conjugate(z0): zb0, sp.conjugate(z1): zb1,
        sp.conjugate(ψ0): ψb0, sp.conjugate(ψ1): ψb1,
    }
    expr = expr.subs(subs)
    expr = sp.together(expr)

    num, den = sp.fraction(expr)
    num_red = Grob.reduce(sp.expand(num))[1]
    den_red = Grob.reduce(sp.expand(den))[1]

    expr_red = sp.cancel(num_red/den_red)
    expr_red = expr_red.subs({ε_T**-1: ε_T, ε_P**-1: ε_P, ε_T**2: 1, ε_P**2: 1})

    return sp.simplify(expr_red)

# =====================   VERIFICATION   =====================

def compare(actual, target, check_name, a, b, c):
    """ Raise an error if actual and target don't match """
    diff = sp.simplify(actual - target)
    diff0 = impose_constraints(diff[0])
    diff1 = impose_constraints(diff[1])
    assert diff0 == 0 and diff1 == 0, (diff0, diff1, check_name, a, b, c)

def verify_sigma_identity():
    Pp = sp.diag(1, -1, -1, -1) * p
    lhs = -ε * sp.conjugate(sigma(Pp)) * ε
    rhs = sigma(p)
    delta = sp.simplify(lhs - rhs)
    assert delta == sp.zeros(2), delta

def verify_relations():
    for double_cover in range(2**3):
        a, b, c = map(int, format(double_cover, '03b'))

        P = partial(act_P,             a=a, b=b, c=c)
        T = partial(act_T,             a=a, b=b, c=c)
        G = partial(act_gauge_transfo, a=a, b=b, c=c)

        checks = [
            ('G squared',  (-1)**a *     z, compose(G, G)),
            ('P squared',  (-1)**a *     z, compose(P, P)),
            ('T squared',  (-1)**b *     z, compose(T, T)),
            ('PT squared', (-1)**c *     z, compose(P, T, P, T)),
            ('PG squared',               z, compose(P, G, P, G)),
            ('TG squared', (-1)**(a+b) * z, compose(T, G, T, G)),
            ('PG', -(-1)**a * ε_P * sp.sign(p[0])**(a+b+c) * R*sp.conjugate(w(p, z, ε_P)), compose(P, G)),
        ]
        for check_name, z_target, func in checks:
            _, z_actual, _ = func(p, z, ε_P)
            compare(z_actual, z_target, check_name, a, b, c)

        commutation_checks = [
            ('P and G commute', compose(P, G), compose(G, P)),
            ('T and G commute', compose(T, G), compose(G, T)),
        ]
        for check_name, func1, func2 in commutation_checks:
            _, z_1, _ = func1(p, z, ε_P)
            _, z_2, _ = func2(p, z, ε_P)
            compare(z_1, z_2, check_name, a, b, c)

        inner_product_checks = [
            ('Inner product invariant under PG', compose(P, G)),
            ('Inner product invariant under T', T),
        ]
        for check_name, func in inner_product_checks:
            ψ_prime, z_prime = func(p, ψ, ε_P)[1], func(p, z, ε_P)[1]
            diff = (ψ_prime.T * ε * z_prime)[0] - (ψ.T * ε * z)[0]
            diff = impose_constraints(sp.simplify(diff))
            assert diff==0, diff

        compute_time_elapsed(f"Sqaure, commutation, and inner product relations verified for: a={a}, b={b}, c={c}")

if __name__ == "__main__":
    main()

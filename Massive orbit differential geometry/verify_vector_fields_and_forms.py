from performance_benchmarking import compute_time_elapsed

from massive_orbits_dg import (
    C, sp, η, ε, 
    dot, sigma, decompose, apply_vf, simplify_onshell, simplify_vf, construct_Lambda_cols,
    I, half, m, s, ε_P, ε_T,
    p0f, p1f, p2f, p3f, z0f, z1f, zb0f, zb1f,
    e_p, e_q, e_z, e_zb,
    dp, dq, dz, dzb,
    w, wb, z, zb, λ,
)

VERBOSE = False # Prints time to execute each test

Λ0, Λ1, Λ2, Λ3 = construct_Lambda_cols()

# ---------------------------------------------------------
# Coordinate conventions fixed by the manuscript, not duality
# ---------------------------------------------------------

def check_coordinate_conventions():
    # z_A w^A = ε_P s, with the index lowered on the right.
    assert simplify_onshell(dot(z, ε*w) - ε_P*s) == 0
    # λ has columns scaled by sqrt(s), hence det(λ/sqrt(s)) = 1.
    assert simplify_onshell(λ.det() - s) == 0

    Λ = sp.Matrix.hstack(Λ0, Λ1, Λ2, Λ3)
    for ep in (1, -1):
        for et in (1, -1):
            # λ = identity, with each of the four discrete group labels.
            zs = sp.sqrt(s) * sp.Matrix([1, 0] if ep == 1 else [0, 1])
            ws = sp.sqrt(s) * sp.Matrix([0, 1] if ep == 1 else [1, 0])
            canonical = {
                p0f: et*m, p1f: 0, p2f: 0, p3f: 0,
                z0f: zs[0], z1f: zs[1], zb0f: zs[0], zb1f: zs[1],
                ε_P: ep, ε_T: et,
            }
            assert w.subs(canonical).applyfunc(sp.simplify) == ws
            assert wb.subs(canonical).applyfunc(sp.simplify) == ws
            assert (λ/sp.sqrt(s)).subs(canonical).applyfunc(sp.simplify) == sp.eye(2)
            assert Λ.subs(canonical).applyfunc(sp.simplify) == sp.diag(et, ep, ep, ep)

# -------------------------------------------
# Candidate left invariant vector fields
# -------------------------------------------

V_T0 = dot(Λ0, e_q)
V_T1 = dot(Λ1, e_q)
V_T2 = dot(Λ2, e_q)
V_T3 = dot(Λ3, e_q)

V_J1 =                     (-I)     *( dot(w, e_z) - dot(wb, e_zb) ) * half
V_J2 =                     (ε_P)    *( dot(w, e_z) + dot(wb, e_zb) ) * half
V_J3 =                     (-I*ε_P) *( dot(z, e_z) - dot(zb, e_zb) ) * half

V_K1 = -m * dot(Λ1, e_p) + (-ε_P*ε_T)*( dot(w, e_z) + dot(wb, e_zb) ) * half
V_K2 = -m * dot(Λ2, e_p) + (-I*ε_T)  *( dot(w, e_z) - dot(wb, e_zb) ) * half
V_K3 = -m * dot(Λ3, e_p) + (-ε_T)    *( dot(z, e_z) + dot(zb, e_zb) ) * half

VFs = {
    "V_T0": V_T0, "V_T1": V_T1, "V_T2": V_T2, "V_T3": V_T3,
    "V_J1": V_J1, "V_J2": V_J2, "V_J3": V_J3,
    "V_K1": V_K1, "V_K2": V_K2, "V_K3": V_K3,
}

def check_vector_field_decomposition():
    # Catch dropped components before testing their derivatives or brackets.
    # In particular, old SymPy versions can lose the momentum part of V_K3
    # when extracting coefficients from an entire noncommutative sum.
    for name, vector in VFs.items():
        reconstructed = sum(coef * bv for bv, coef in decompose(vector))
        defect = sp.expand(reconstructed - vector)
        assert defect == 0, ("vector-field decomposition", name, defect)


# -------------------
# Tangency checks
# -------------------

# p_sq_f = p_μ p^μ = m^2
p_sq_f = p0f**2 - p1f**2 - p2f**2 - p3f**2
# pzzb = p_{A\dotA} z^A zb^\dotA = sm/sqrt2
pzzb_f = ε_T * (sp.Matrix([z0f, z1f]).T * ε.T * sigma(p0f,p1f,p2f,p3f) * ε * sp.Matrix([zb0f, zb1f]))[0]

def check_tangency():
    for VF_name, VF in VFs.items():
        print(f"Tangency test for {VF_name}")

        tangent_p_sq = simplify_onshell(apply_vf(VF, p_sq_f))
        tangent_pzzb = simplify_onshell(apply_vf(VF, pzzb_f))

        if VERBOSE: compute_time_elapsed("Tangency verified")

        assert tangent_p_sq==0, ("p_sq test", VF_name, tangent_p_sq)
        assert tangent_pzzb==0, ("pzzb test", VF_name, tangent_pzzb)

# ------------------------------
# Commutation relation chekcs
# ------------------------------

tests = {
    # --- Lorentz subalgebra ---
    "[V_J1, V_J2] ==  V_J3":  (V_J1, V_J2,  V_J3),
    "[V_J2, V_J3] ==  V_J1":  (V_J2, V_J3,  V_J1),
    "[V_J3, V_J1] ==  V_J2":  (V_J3, V_J1,  V_J2),

    "[V_J1, V_K2] ==  V_K3":  (V_J1, V_K2,  V_K3),
    "[V_J2, V_K3] ==  V_K1":  (V_J2, V_K3,  V_K1),
    "[V_J3, V_K1] ==  V_K2":  (V_J3, V_K1,  V_K2),

    "[V_J2, V_K1] == -V_K3":  (V_J2, V_K1, -V_K3),
    "[V_J3, V_K2] == -V_K1":  (V_J3, V_K2, -V_K1),
    "[V_J1, V_K3] == -V_K2":  (V_J1, V_K3, -V_K2),

    "[V_K1, V_K2] == -V_J3":  (V_K1, V_K2, -V_J3),
    "[V_K2, V_K3] == -V_J1":  (V_K2, V_K3, -V_J1),
    "[V_K3, V_K1] == -V_J2":  (V_K3, V_K1, -V_J2),

    "[V_J1, V_K1] == 0":      (V_J1, V_K1, 0),
    "[V_J2, V_K2] == 0":      (V_J2, V_K2, 0),
    "[V_J3, V_K3] == 0":      (V_J3, V_K3, 0),

    # --- Translations commute ---
    "[V_T0, V_T1] == 0":      (V_T0, V_T1, 0),
    "[V_T0, V_T2] == 0":      (V_T0, V_T2, 0),
    "[V_T0, V_T3] == 0":      (V_T0, V_T3, 0),
    "[V_T1, V_T2] == 0":      (V_T1, V_T2, 0),
    "[V_T2, V_T3] == 0":      (V_T2, V_T3, 0),
    "[V_T3, V_T1] == 0":      (V_T3, V_T1, 0),

    # --- Rotation action on translations ---
    "[V_J1, V_T0] == 0":      (V_J1, V_T0, 0),
    "[V_J2, V_T0] == 0":      (V_J2, V_T0, 0),
    "[V_J3, V_T0] == 0":      (V_J3, V_T0, 0),

    "[V_J1, V_T1] == 0":      (V_J1, V_T1, 0),
    "[V_J1, V_T2] ==  V_T3":  (V_J1, V_T2,  V_T3),
    "[V_J1, V_T3] == -V_T2":  (V_J1, V_T3, -V_T2),

    "[V_J2, V_T1] == -V_T3":  (V_J2, V_T1, -V_T3),
    "[V_J2, V_T2] == 0":      (V_J2, V_T2, 0),
    "[V_J2, V_T3] ==  V_T1":  (V_J2, V_T3,  V_T1),

    "[V_J3, V_T1] ==  V_T2":  (V_J3, V_T1,  V_T2),
    "[V_J3, V_T2] == -V_T1":  (V_J3, V_T2, -V_T1),
    "[V_J3, V_T3] == 0":      (V_J3, V_T3, 0),

    # --- Boost action on translations ---
    "[V_K1, V_T0] == -V_T1":  (V_K1, V_T0, -V_T1),
    "[V_K1, V_T1] == -V_T0":  (V_K1, V_T1, -V_T0),
    "[V_K1, V_T2] == 0":      (V_K1, V_T2, 0),
    "[V_K1, V_T3] == 0":      (V_K1, V_T3, 0),

    "[V_K2, V_T0] == -V_T2":  (V_K2, V_T0, -V_T2),
    "[V_K2, V_T1] == 0":      (V_K2, V_T1, 0),
    "[V_K2, V_T2] == -V_T0":  (V_K2, V_T2, -V_T0),
    "[V_K2, V_T3] == 0":      (V_K2, V_T3, 0),

    "[V_K3, V_T0] == -V_T3":  (V_K3, V_T0, -V_T3),
    "[V_K3, V_T1] == 0":      (V_K3, V_T1, 0),
    "[V_K3, V_T2] == 0":      (V_K3, V_T2, 0),
    "[V_K3, V_T3] == -V_T0":  (V_K3, V_T3, -V_T0),
}

def commutator(X, Y):
    Xd, Yd = dict(decompose(X)), dict(decompose(Y))
    out = 0
    for bv in C.base_vectors():
        X_bv_c, Y_bv_c = Xd.get(bv,0), Yd.get(bv,0)
        out += (apply_vf(X, Y_bv_c) - apply_vf(Y, X_bv_c)) * bv
    return sp.expand(out)

def check_commutation_relations(total_time_elapsed=0):
    for test_name, (V1, V2, V3) in tests.items():
        print(f"Testing: {test_name}")
        defect = simplify_vf(commutator(V1, V2) - V3)
        if VERBOSE: compute_time_elapsed("Commutator computed")
        assert defect==0, (test_name, defect)
    return total_time_elapsed

# ---------------------------------------
# Candidate left invariant one forms
# ---------------------------------------

Θ_T0 = η[0,0]*dot(η*Λ0, dq)
Θ_T1 = η[1,1]*dot(η*Λ1, dq)
Θ_T2 = η[2,2]*dot(η*Λ2, dq)
Θ_T3 = η[3,3]*dot(η*Λ3, dq)

Θ_J1 = (-ε_T/m)*dot(η*Λ2, dp) + (I*ε_P/s)*( dot(z, ε*dz) - dot(zb, ε*dzb))
Θ_J2 = (ε_T/m)*dot(η*Λ1, dp)  + (1/s)     *( dot(z, ε*dz) + dot(zb, ε*dzb))
Θ_J3 =                          (-I/s)    *( dot(w, ε*dz) - dot(wb, ε*dzb))

Θ_K1 = (1/m)*dot(η*Λ1, dp)
Θ_K2 = (1/m)*dot(η*Λ2, dp)
Θ_K3 = (1/m)*dot(η*Λ3, dp)

forms = {
    "Θ_T0": Θ_T0, "Θ_T1": Θ_T1, "Θ_T2": Θ_T2, "Θ_T3": Θ_T3, 
    "Θ_J1": Θ_J1, "Θ_J2": Θ_J2, "Θ_J3": Θ_J3,
    "Θ_K1": Θ_K1, "Θ_K2": Θ_K2, "Θ_K3": Θ_K3, 
}

def check_duality():
    for form_name, form in forms.items():
        for VF_name, VF in VFs.items():
            contraction = simplify_onshell(form.rcall(VF))
            
            same_generators = (form_name[2:] == VF_name[2:])
            if same_generators:
                print(f"Verifying {form_name}({VF_name})==1")
                assert contraction==1, (form_name, VF_name, contraction)
            else:
                print(f"Verifying {form_name}({VF_name})==0")
                assert contraction==0, (form_name, VF_name, contraction)    

# --------
# Main
# --------

if __name__ == "__main__":
    check_coordinate_conventions()
    compute_time_elapsed("Spinor normalization and canonical group coordinates verified")
    check_vector_field_decomposition()
    compute_time_elapsed("Vector-field component reconstruction verified")
    check_tangency()
    compute_time_elapsed("Tangency for vector fields verified")
    check_commutation_relations()
    compute_time_elapsed("Commutation relations for vector fields verified")
    check_duality()
    compute_time_elapsed("One forms verified")

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from performance_benchmarking import compute_time_elapsed

from massless_orbits_dg import (
    C, sp, η, ε, 
    dot, sigma, decompose, apply_vf, simplify_onshell, simplify_vf, construct_Lambda_cols,
    I, half, h, ε_P, ε_T,
    π, πb, ω, ωb,
    e_π, e_πb, e_ω, e_ωb, e_q,
    dπ, dπb, dω, dωb, dq,
)

VERBOSE = False # Prints time to execute each commutation check

Λ0, Λ1, Λ2, Λ3 = construct_Lambda_cols()

# -------------------------------------------
# Candidate left invariant vector fields
# -------------------------------------------

V_T0 = dot(Λ0, e_q)
V_T1 = dot(Λ1, e_q)
V_T2 = dot(Λ2, e_q)
V_T3 = dot(Λ3, e_q)

V_N1p = (-ε_P*ε_T) * (              + dot(π, e_ω)                 + dot(πb, e_ωb) )
V_N1m = (-ε_P*ε_T) * ( dot(ω, e_π)                + dot(ωb, e_πb)                 )

V_N2p = (I)        * (              + dot(π, e_ω)                 - dot(πb, e_ωb) )
V_N2m = (-I)       * ( dot(ω, e_π)                - dot(ωb, e_πb)                 )

V_J3 = (-I*ε_P*ε_T) * ( dot(π, e_π) - dot(ω, e_ω) - dot(πb, e_πb) + dot(ωb, e_ωb) ) * half
V_K3 = (-1)         * ( dot(π, e_π) - dot(ω, e_ω) + dot(πb, e_πb) - dot(ωb, e_ωb) ) * half

VFs = {
    "V_T0": V_T0, "V_T1": V_T1, "V_T2": V_T2, "V_T3": V_T3,
    "V_N1+": V_N1p, "V_N1-": V_N1m, "V_N2+": V_N2p, "V_N2-": V_N2m, "V_J3": V_J3, "V_K3": V_K3,
}

# -------------------
# Tangency checks
# -------------------

inner_prod   = ( π.T * ε * ω )[0]
inner_prod_B = ( πb.T * ε * ωb )[0]

def check_tangency():
    for VF_name, VF in VFs.items():
        print(f"Tangency test for {VF_name}")

        tangent   = simplify_onshell(apply_vf(VF, inner_prod))
        tangent_B = simplify_onshell(apply_vf(VF, inner_prod_B))

        if VERBOSE: compute_time_elapsed("Tangency verified")

        assert tangent==0,   ("tangent test"    , VF_name, tangent)
        assert tangent_B==0, ("tangent bar test", VF_name, tangent_B)

# ------------------------------
# Commutation relation chekcs
# ------------------------------

tests = {
    # --- Lorentz subalgebra ---
    "[V_N1+, V_J3] == -V_N2+": (V_N1p, V_J3, -V_N2p),
    "[V_N2+, V_J3] ==  V_N1+": (V_N2p, V_J3,  V_N1p),
    "[V_N1+, V_K3] ==  V_N1+": (V_N1p, V_K3,  V_N1p),
    "[V_N2+, V_K3] ==  V_N2+": (V_N2p, V_K3,  V_N2p),
    "[V_N1+, V_N2+] == 0"    : (V_N1p, V_N2p, 0),

    "[V_N1-, V_J3] == -V_N2-": (V_N1m, V_J3, -V_N2m),
    "[V_N2-, V_J3] ==  V_N1-": (V_N2m, V_J3,  V_N1m),
    "[V_N1-, V_K3] == -V_N1-": (V_N1m, V_K3, -V_N1m),
    "[V_N2-, V_K3] == -V_N2-": (V_N2m, V_K3, -V_N2m),
    "[V_N1-, V_N2-] == 0"    : (V_N1m, V_N2m, 0),

    "[V_N1+, V_N2-] == -2 V_J3": (V_N1p, V_N2m, -2*V_J3),
    "[V_N1-, V_N2+] == -2 V_J3": (V_N1m, V_N2p, -2*V_J3),
    "[V_N1+, V_N1-] == -2 V_K3": (V_N1p, V_N1m, -2*V_K3),
    "[V_N2+, V_N2-] == -2 V_K3": (V_N2p, V_N2m, -2*V_K3),

    "[V_J3, V_K3] == 0":  (V_J3, V_K3, 0),

    # --- Lorentz action on translations ---
    "[V_J3, V_T0] == 0":      (V_J3, V_T0, 0),
    "[V_J3, V_T1] ==  V_T2":  (V_J3, V_T1,  V_T2),
    "[V_J3, V_T2] == -V_T1":  (V_J3, V_T2, -V_T1),
    "[V_J3, V_T3] == 0":      (V_J3, V_T3, 0),

    "[V_K3, V_T0] ==  V_T3":  (V_K3, V_T0, -V_T3),
    "[V_K3, V_T1] == 0":      (V_K3, V_T1, 0),
    "[V_K3, V_T2] == 0":      (V_K3, V_T2, 0),
    "[V_K3, V_T3] ==  V_T0":  (V_K3, V_T3, -V_T0),

    "[V_N1+, V_T0 + V_T3] == 0": (V_N1p, V_T0 + V_T3, 0),
    "[V_N1-, V_T0 - V_T3] == 0": (V_N1m, V_T0 - V_T3, 0),
    "[V_N2+, V_T0 + V_T3] == 0": (V_N2p, V_T0 + V_T3, 0),
    "[V_N2-, V_T0 - V_T3] == 0": (V_N2m, V_T0 - V_T3, 0),
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

# ---------------------------------------
# Candidate left invariant one forms
# ---------------------------------------

Θ_T0 = η[0,0]*dot(η*Λ0, dq)
Θ_T1 = η[1,1]*dot(η*Λ1, dq)
Θ_T2 = η[2,2]*dot(η*Λ2, dq)
Θ_T3 = η[3,3]*dot(η*Λ3, dq)

Θ_N1p =              (              + dot(ω, ε*dω)                  + dot(ωb, ε*dωb) ) * half
Θ_N1m = (-1)*        ( dot(π, ε*dπ)                + dot(πb, ε*dπb)                  ) * half

Θ_N2p = (I*ε_P*ε_T)* (              + dot(ω, ε*dω)                  - dot(ωb, ε*dωb) ) * half
Θ_N2m = (I*ε_P*ε_T)* ( dot(π, ε*dπ)                - dot(πb, ε*dπb)                  ) * half

Θ_J3 = (-I)*         ( dot(ω, ε*dπ) + dot(π, ε*dω) - dot(ωb, ε*dπb) - dot(πb, ε*dωb) ) * half
Θ_K3 = (ε_P*ε_T)*    ( dot(ω, ε*dπ) + dot(π, ε*dω) + dot(ωb, ε*dπb) + dot(πb, ε*dωb) ) * half

forms = {
    "Θ_T0": Θ_T0, "Θ_T1": Θ_T1, "Θ_T2": Θ_T2, "Θ_T3": Θ_T3, 
    "Θ_N1+": Θ_N1p, "Θ_N1-": Θ_N1m, "Θ_N2+": Θ_N2p, "Θ_N2-": Θ_N2m, "Θ_J3": Θ_J3, "Θ_K3": Θ_K3,
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
    check_tangency()
    compute_time_elapsed("Tangency for vector fields verified")
    check_commutation_relations()
    compute_time_elapsed("Commutation relations for vector fields verified")
    check_duality()
    compute_time_elapsed("One forms verified")

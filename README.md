# Code-to-paper map

This file summarizes which scripts in this repository verify which calculations in the paper
`Bridging Quantization and Representation Theory - The Orbit Method for the Poincaré Group`.

## Massive-orbit differential geometry

- `Massive orbit differential geometry/verify_vector_fields_and_forms.py`
  Verifies the massive-orbit left-invariant vector fields and dual 1-forms in **Table 2** in section **2.3.2**. This verification of the vector fields is done by computing the Lie brackets between the vector fields and checking they satisfy the Poincaré commutation relations.

- `Massive orbit differential geometry/verify_pre_symplectic_form.py`
  Verifies equation **(2.3.36)** in Proposition **2.21** for the massive pre-symplectic potential written in $(p,q,\theta,\phi)$ coordinates.

- `Massive orbit differential geometry/massive_orbits_dg.py`
  Helper module which sets up the massive coordinate system, on-shell simplification rules, the spinor $w(p,z,ε_P,ε_T)$, and the Lorentz-frame columns $\Lambda$.

## Massless-orbit differential geometry

- `Massless orbit differential geometry/verify_vector_fields_and_forms.py`
  Verifies the massless left-invariant vector fields and dual 1-forms in **Table 3**, section **2.4.2**. As in the massive case, it also checks the commutation relations realized by the vector fields.

- `Massless orbit differential geometry/massless_orbits_dg.py`
  Helper module which sets up the massless coordinate system used in section **2.4.2**.

## Poincare Lie algebra
- `poincare_algebra.py`
  Verifies the concrete Poincaré Lie algebra commutation relations in section **2.1**. Starting from the definitions **(2.1.7)**, **(2.1.8)** and **(2.1.10)**, the script then verifies equations **(2.1.9)** and **(2.1.11)**. This is the algebraic backbone used later in both the massive and massless orbit calculations.

## Vector and spinor identities

- `pauli_lubanski_vector.py`
  Verifies several spin/momentum identities used in sections **2.2** and **2.3.1**:
  - the Pauli-Lubanski vectors in equation **(2.2.8)** for the reference dual Lie algebra elements,
  - equation **(2.3.38)**: the 4-vector corresponding to $\sqrt{p}$,
  - eqaution **(2.3.28)**: the 3-vector s corresponding to the rest frame spinor u,
  - equation **(2.3.29)**: the Pauli-Lubanski pseudovector W expressed in terms of p and s.

- `hodge_dual.py`
  Verifies tensor and spinor identities used in the appendices and in the derivation of the massive spin formulas:
  - equation **(2.2.11)**: the Lorentzian Hodge-dual identity $\ast(p \wedge \ast(p \wedge W)) = \eta(p,p)W - \eta(p,W)p$,
  - equation **(A.27)**: the spinor form of the Levi-Civita tensor,
  - and the that the hodge star eigenvalue of $\Phi_{(AB)} \epsilon_{\dot{A}\dot{B}}$ is -i.
  The latter two are directly relevant to the proof machinery behind **Lemma A.8**, which proves equation **(2.3.8)**.

- `spinor_identity.py`
  Verifies eqation **(2.4.20)** which is an auxiliary Pauli-matrix / epsilon-tensor identity used to compute the differential form which defines the measure for the Hermitian inner product for massless representations.

## Discrete symmetries and double covers

- `discrete_transformations.py`
  Verifies the parity, time-reversal (derived in **Lemma 2.19**), and gauge-transformation formulas (derived in proof of **Theorem 2.23**) in the $(p,z,\epsilon)$ variables. In particular, it checks the square/commutation relations of the discrete generators. It also checks the gauge-fixed parity action that feeds into the parity-equivariance discussion in equation **(2.4.10)**.

## Action principle

- `momentum_EL_equations.py`
  Starts from the non-manifestly covariant massive action in equation **(2.3.44)** and verifies the momentum Euler-Lagrange equations leading to the hidden-momentum relation **(2.3.46)** in section **2.3.4**.

- `MPD_equations.py`
  This is the script that connects the action-principle calculation to the Mathisson-Papapetrou-Dixon form quoted in the paper. It uses the Thomas-precession equation **(2.3.45)** and the hidden-momentum relation **(2.3.46)** to verify the covariant spin evolution equation **(2.3.47)** in section **2.3.4**.

## Passing from momentum space to position space

- `dirac_equation.py`
  Verifies the claim at the end of section **2.4.1** that the Fourier-transformed spin-`1/2` wavefunction in eqauation **(2.4.16)** satisfies the Dirac equation in the chiral basis.

## Files with a smaller or indirect role

- `performance_benchmarking.py`
  Only reports elapsed time for the symbolic checks. It does not correspond to a paper calculation by itself.

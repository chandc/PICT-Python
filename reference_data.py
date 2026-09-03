"""Published benchmark values for the square cylinder at low Reynolds number.

SOURCE. A. Sohankar, C. Norberg and L. Davidson, "Low-Reynolds-number flow around a square
cylinder at incidence: study of blockage, onset of vortex shedding and outlet boundary
condition", Int. J. Numer. Meth. Fluids 26:39-56 (1998).

WHY THIS ONE. It is the only source found that reports Re = 100 at zero incidence together with
the BLOCKAGE it was computed at, and it varies blockage, grid and outlet condition separately,
so its own spread can be read off rather than guessed. Our grid is 5% blockage, which is exactly
their case 5 of Table III -- a like-for-like comparison rather than a band remembered from
somewhere.

WHAT WE HAD BEFORE WAS THE WRONG COMPARISON. sqcyl_onset.py carried RE_C_REF = (45, 47), which
is the ZERO-blockage experimental estimate (Norberg, quoted in this paper as 47 +/- 2). At 5%
blockage the same paper computes Re_cr = 51.2 +/- 1.0 and states that the critical Reynolds
number increases with blockage. Comparing a 5%-blockage result against a zero-blockage reference
made a correct answer look 10% wrong -- the same trap as section 1 of measurement_traps.md.

NOTATION. C_L' is the rms lift. C_pb is the base pressure coefficient, tabulated as -C_pb, so
their 0.661 means C_p = -0.661 on the base. C_ps is the stagnation pressure coefficient, and it
is NOT 1 at finite blockage: they report 1.052 at 5% and 1.083 at 2.5%.
"""

CITATION = ("Sohankar, Norberg & Davidson, Int. J. Numer. Meth. Fluids 26:39-56 (1998)")

# Table III -- blockage effect at alpha = 0
BLOCKAGE = {
    #  beta%      grid        St     C_D    C_Dp    C_L'   -C_pb   C_ps
    100: [(5.0, "96x94", 0.146, 1.460, 1.414, 0.139, 0.661, 1.052),
          (2.5, "96x132", 0.144, 1.444, 1.399, 0.141, 0.611, 1.083)],
}

# Table IV -- grid refinement at Re = 100, alpha = 0, beta = 5%
REFINEMENT_RE100 = [
    # Delta   N_b   BC     St     C_D    C_Dp    C_L'   -C_pb   C_ps
    (0.50, 20, "CBC", 0.146, 1.460, 1.414, 0.139, 0.661, 1.052),
    (0.30, 20, "NBC", 0.146, 1.477, 1.433, 0.156, None, None),
    (0.15, 40, "CBC", 0.146, 1.478, 1.434, 0.153, 0.678, 1.059),
]

# Table II -- outlet boundary condition at Re = 100, alpha = 0
OUTLET_RE100 = [
    # BC     X_d     St     C_D    C_Dp    C_L'   -C_pb   C_ps
    ("NBC", 26, 0.147, 1.464, 1.418, 0.138, 0.663, 1.052),
    ("NBC", 10, 0.150, 1.491, 1.441, 0.024, 0.691, 1.051),
    ("CBC", 10, 0.146, 1.460, 1.414, 0.139, 0.661, 1.052),
    ("CBC", 5.5, 0.145, 1.452, 1.407, 0.146, 0.661, 1.053),
    ("CBC", 3, 0.133, 1.426, 1.382, 0.148, 0.616, 1.052),
]

# Table V -- critical Reynolds number, and the text around it
RE_CR = {"alpha0_beta5": (51.2, 1.0),          # this paper, 5% blockage
         "zero_blockage_experiment": (47.0, 2.0),   # Norberg, quoted therein
         "kelkar_patankar_beta14.2": 53.0}

# Recirculation length behind the body, from the text: L_r = 3.69, 2.20, 1.67 at Re = 50, 100, 200
L_R = {50: 3.69, 100: 2.20, 200: 1.67}


def re100_beta5():
    """The single directly comparable case: Re = 100, alpha = 0, blockage 5%."""
    b, grid, St, CD, CDp, CL, mCpb, Cps = BLOCKAGE[100][0]
    return dict(St=St, C_D=CD, C_Dp=CDp, C_L_rms=CL, C_p_base=-mCpb, C_p_stag=Cps,
                blockage=b, grid=grid, L_r=L_R[100], source=CITATION)


if __name__ == "__main__":
    d = re100_beta5()
    print(f"  {CITATION}\n")
    print(f"  Re = 100, alpha = 0, blockage {d['blockage']}%, grid {d['grid']}:")
    for k in ("St", "C_D", "C_Dp", "C_L_rms", "C_p_base", "C_p_stag", "L_r"):
        print(f"    {k:<10} {d[k]:+.4f}")
    print(f"\n  their own spread across three grids (Table IV): "
          f"C_L' {min(r[6] for r in REFINEMENT_RE100):.3f}-"
          f"{max(r[6] for r in REFINEMENT_RE100):.3f}, "
          f"C_D {min(r[4] for r in REFINEMENT_RE100):.3f}-"
          f"{max(r[4] for r in REFINEMENT_RE100):.3f}, St all 0.146")
    lo, hi = RE_CR["alpha0_beta5"]
    print(f"  Re_cr at 5% blockage = {lo} +/- {hi};  zero-blockage experiment "
          f"{RE_CR['zero_blockage_experiment'][0]} +/- {RE_CR['zero_blockage_experiment'][1]}")

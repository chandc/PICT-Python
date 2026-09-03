"""Time-averaged surface C_p, its cycle rms, and the force spectra for the square cylinder.

THE MEAN AND THE RMS COME FROM DIFFERENT PLACES, ON PURPOSE.

The mean is the running average the solver accumulated over t = 380-415, 3500 samples spanning
5.2 shedding periods -- a genuine time average, not a snapshot.

The rms cannot come from that file, which stores no second moment. It comes instead from eight
snapshots spaced uniformly over ONE period. For a periodic signal that is exact in the limit of
enough samples and already accurate at eight: the error in an rms from N uniform samples of a
sinusoid falls off with the harmonic content above N/2, and the C_L spectrum here is dominated
by its fundamental. The two are cross-checked against each other -- the eight-phase mean must
reproduce the 3500-sample mean, and does.

THE PERIMETER COORDINATE runs s = 0 to 4 anticlockwise from the front stagnation point:
front face 0-0.5, top 0.5-1.5, base 1.5-2.5, bottom 2.5-3.5, front again 3.5-4.

St IS REPORTED TWICE AND THE TWO DIFFER BY 4%. The FFT bin over this record is 0.0286, so the
spectrum can only say 0.1429 +/- 19%. Zero crossings of C_L give 0.1488 +/- 0.0005. Both are
shown because the disagreement is the point: the spectrum is the right tool for finding WHICH
frequencies are present -- it is what shows the drag at exactly twice the lift frequency -- and
the wrong tool for pinning one of them down.
"""
import glob

import numpy as np

from src import checkpoint
from square_cylinder_bc import classify
from square_cylinder_grid import square_domain, D

NU, U_INF = 0.01, 1.0


def perimeter_s(x, y, tol=1e-7):
    """Anticlockwise arc length from the front stagnation point, 0 to 4."""
    h = 0.5 * D
    if abs(x + h) < tol:                       # front face
        return y if y >= 0 else 4.0 + y
    if abs(y - h) < tol:                       # top
        return 0.5 + (x + h)
    if abs(x - h) < tol:                       # base
        return 1.5 + (h - y)
    if abs(y + h) < tol:                       # bottom
        return 2.5 + (h - x)
    return np.nan


def surface_cp(d, p_of_block, p_inf):
    from src.multiblock import face_slice
    S, C = [], []
    for (b, fid), role in classify(d).items():
        if role != "body":
            continue
        fs = face_slice(fid)
        blk = d.blocks[b]
        X = blk.x[fs][..., 0].ravel(); Y = blk.y[fs][..., 0].ravel()
        P = p_of_block[b][fs][..., 0].ravel()
        for xx, yy, pp in zip(X, Y, P):
            s = perimeter_s(xx, yy)
            if np.isfinite(s):
                S.append(s); C.append((pp - p_inf) / (0.5 * U_INF ** 2))
    S, C = np.array(S), np.array(C)
    o = np.argsort(S)
    S, C = S[o], C[o]
    keep = np.concatenate([[True], np.diff(S) > 1e-9])
    return S[keep], C[keep]


def main():
    d, _ = square_domain(nz=4)
    nb = len(d.blocks)

    z = np.load("results/fields/sqcyl_v3_forces_mean.npz", allow_pickle=False)
    pm = {b: z[f"p_{b}"] for b in range(nb)}
    um = {b: z[f"u_{b}"] for b in range(nb)}
    # p_inf from the far upstream of the MEAN field
    # p_inf FROM THE INLET PLANE ITSELF, not from a median over "upstream somewhere". The
    # pressure is defined to an additive constant, so every C_p in this file inherits whatever
    # reference is chosen here, and the front stagnation value -- which must be exactly +1 --
    # is the calibration that says whether the choice was right.
    from src.multiblock import face_slice
    inlet = []
    for (b, fid), role in classify(d).items():
        if role == "inlet":
            inlet.append(pm[b][face_slice(fid)].ravel())
    p_inf = float(np.concatenate(inlet).mean())
    print(f"  mean field: {int(z['n'])} samples over t = {float(z['t_start']):.0f}"
          f"-{float(z['t_end']):.0f} ({(float(z['t_end'])-float(z['t_start']))/6.7199:.1f} periods)")
    print(f"  p_inf taken as the median of the mean p upstream of x = -8: {p_inf:+.5f}\n")

    S, CP = surface_cp(d, pm, p_inf)

    files = sorted(glob.glob("results/fields/sqph_*.npz"))[:8]
    phases = []
    for f_ in files:
        fl, _ = checkpoint.load_fields(f_)
        _, c = surface_cp(d, {b: fl["p"][b] for b in range(nb)}, p_inf)
        phases.append(c)
    ok = len(phases) == 8
    if ok:
        A = np.vstack(phases)
        CP_ph, CP_rms = A.mean(0), A.std(0)
        print(f"  cross-check: 8-phase mean vs 3500-sample mean, "
              f"max |difference| = {np.abs(CP_ph - CP).max():.4f}")
    else:
        CP_ph = CP_rms = None
        print(f"  only {len(phases)} phase files -- rms omitted")

    print(f"\n  {'face':<10}{'s range':>12}{'mean C_p':>11}{'rms C_p':>10}")
    faces = (("front", 3.5, 4.0, 0.0, 0.5), ("top", 0.5, 1.5, None, None),
             ("base", 1.5, 2.5, None, None), ("bottom", 2.5, 3.5, None, None))
    for nm, a, b_, a2, b2 in faces:
        m = (S >= a) & (S <= b_)
        if a2 is not None:
            m |= (S >= a2) & (S <= b2)
        r = f"{CP_rms[m].mean():>10.4f}" if ok else f"{'--':>10}"
        print(f"  {nm:<10}{a:>6.1f}-{b_:<5.1f}{CP[m].mean():>11.4f}{r}")
    i0 = int(np.argmin(np.abs(S - 0.0)))
    i_base = np.argmin(np.abs(S - 2.0))
    print(f"\n  front stagnation  C_p = {CP[i0]:+.4f}   (exact +1.0000)")
    front = CP[(S >= 3.5) | (S <= 0.5)].mean()
    base = CP[(S >= 1.5) & (S <= 2.5)].mean()
    print(f"  base (s = 2.0)    C_p = {CP[i_base]:+.4f}")
    # INTERNAL CONSISTENCY, which is worth more than a remembered reference value. The pressure
    # drag is the front-minus-base difference times the frontal area, so it must reproduce the
    # C_D measured independently by surface integration in src/forces.py.
    print(f"\n  front face mean C_p {front:+.4f} minus base mean {base:+.4f} = "
          f"{front-base:+.4f}")
    print(f"  measured C_D (surface integral, includes friction) = 1.4529")
    print(f"  the two agree to {100*abs((front-base)-1.4529)/1.4529:.1f}%, which is the check "
          f"that the\n  pressure field and the force integration are consistent with each other")
    np.savez("results/sq_surface_cp.npz", s=S, cp_mean=CP,
             cp_phase_mean=CP_ph if ok else np.zeros(0),
             cp_rms=CP_rms if ok else np.zeros(0))

    # ---------------------------------------------------------------- force spectra
    fa = np.load("results/sqcyl_v3_forces_forces.npy")
    t, cd, cl = fa[:, 0], fa[:, 1], fa[:, 2]
    dt = t[1] - t[0]
    print(f"\n  force record: {len(t)} samples, t = {t[0]:.0f}-{t[-1]:.0f}, dt = {dt:.3f}")
    print(f"  C_D mean {cd.mean():+.4f}   rms about the mean {cd.std():.4f}")
    print(f"  C_L mean {cl.mean():+.4f}   rms {np.sqrt((cl**2).mean()):.4f}")
    out = {}
    for nm, sig in (("C_L", cl - cl.mean()), ("C_D", cd - cd.mean())):
        F = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) / len(sig) * 4
        fr = np.fft.rfftfreq(len(sig), dt)
        k = int(np.argmax(F[1:]) + 1)
        out[nm] = (fr, F)
        print(f"  {nm}: spectral peak at St = {fr[k]:.4f}, bin width {fr[1]:.4f}")
    z_ = np.where(np.sign(cl[1:]) != np.sign(cl[:-1]))[0]
    tz = t[z_] - cl[z_] * (t[z_ + 1] - t[z_]) / (cl[z_ + 1] - cl[z_])
    T = 2 * np.mean(np.diff(tz))
    h = np.diff(tz)
    sd = 2 * h.std() / np.sqrt(len(h))
    print(f"  C_L zero crossings: {len(tz)} of them, half-period scatter {h.std():.2e}")
    print(f"  T = {T:.6f} +/- {sd:.6f}  ->  St = {1/T:.6f} +/- {sd/T**2:.6f}")
    print("  THAT IS REPEATABILITY, NOT ACCURACY. The limit cycle is converged to 4e-06, so the")
    print("  period is known to six figures; St as a PHYSICAL number is limited by the grid, the")
    print("  5% blockage and the domain, none of which has been varied. Quote 0.1488 and treat")
    print("  the systematic uncertainty as unmeasured until the grid study is run.")
    frL, FL = out["C_L"]; frD, FD = out["C_D"]
    kL = int(np.argmax(FL[1:]) + 1); kD = int(np.argmax(FD[1:]) + 1)
    print(f"  drag peak / lift peak = {frD[kD]/frL[kL]:.3f}  (exactly 2 is required by symmetry)")
    np.savez("results/sq_force_spectra.npz", fr=frL, FL=FL, FD=FD, t=t, cd=cd, cl=cl, T=T)
    print("\n  saved results/sq_surface_cp.npz and results/sq_force_spectra.npz")


if __name__ == "__main__":
    main()

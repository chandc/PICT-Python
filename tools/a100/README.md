# Running the 2.5D unstructured LES solver on an A100 (or any CUDA 12 GPU)

What it is: the unstructured collocated finite-volume solver in the x-y plane, Fourier in a periodic
span, RK3 with a pressure projection per stage, WALE/Smagorinsky, all of it on CuPy
(`src/upiso25.py`, `device="gpu"`). Verified against the CPU path to all printed digits
(reference/skew_unstructured_literature.md sections 51-58). One raw CUDA kernel
(`src/ucuda.py`, CSR times row-major block) compiles at first use through NVRTC, so nothing is
architecture-specific: the same code ran on the GB10 (aarch64) and runs on x86-64 A100/H100.

## Install

Either the container:

    docker build -t upict25 tools/a100/          # nvidia/cuda:12.4 runtime + numpy scipy pyamg cupy-cuda12x
    docker run --gpus all -it -v $PWD:/work upict25

or a venv on a machine with CUDA 12 drivers:

    pip install -r tools/a100/requirements.txt    # cupy-cuda12x is a binary wheel on x86-64; on aarch64 build it or use an NGC image

`pyamg` builds the multigrid hierarchy on the host (needs g++ and python3-dev if no wheel exists).

## Check and profile (first thing to do on the new machine)

    python tools/a100/profile_step.py                  # size ladder on a periodic box, phase split, run-time table
    python tools/a100/profile_step.py --mesh meshes/cylinder_butterfly_fine.msh --nz 64

The profile prints, per size, the whole-step time, the split of one RK stage into nonlinear /
diffusion / momentum solves / pressure solve, and the time per 1e5 cell-modes; then a run-time
table for the target cases at the measured rate. The step is memory-bandwidth bound past a launch
floor of ~70 ms (about 700 kernel launches per step in Python), so a device's expected rate is
the GB10's divided by the bandwidth ratio; measure it rather than trust the scaling.

Correctness on the new device, ~2 minutes:

    TGV_DEVICE=gpu TGV_SOLVER=amg TGV_RE=100 TGV_T=1 TGV_N=32 TGV_NZ=32 python test_utgv3d.py A
    # expect -dE/dt / eps_d = 1.001-1.002 at every sample (section 51), E0 = 31.006277

## Measured on the GB10 (2026-09-25), and what to expect on an A100

`profile_step.py` on the GB10 (whole step, WALE on for the cylinder):

| case | cell-modes | ms/step | ms per 1e5 cell-modes | stage split (ms): nonlinear / momentum x6 / pressure |
|---|---|---|---|---|
| box 128^2 x 64 | 540k | 418 | 77 | 27 / 6 / 50 (8 it) |
| box 256^2 x 64 | 2.2M | 2,340 | 108 | 115 / 50 / 300 (9 it) |
| box 384^2 x 64 | 4.9M | 5,429 | 112 | 247 / 140 / 721 (9 it) |
| **cylinder butterfly-fine 27968 quads x 64, WALE, n_nonorth 3** | 923k | **2,366** | **256** | 168 / 13 / 167 (16 it) |

A real stretched mesh costs 2.3x the box rate: the pressure needs 16 iterations instead of 9 and
the WALE term rides on the nonlinear phase. The step is bandwidth-bound past a ~56 ms launch floor.
**Caveat on the scaling:** the profiler's "nominal bandwidth" comes from the CUDA device
properties, which report 546 GB/s for the GB10 while its LPDDR5X delivers ~273 (the CSR kernel
measured 257 GB/s). Against the real figure an A100-80GB (2039 GB/s) is ~7.5x on the
bandwidth-bound part, not the 3.7x the script prints; take the two as bounds.

| target run | GB10 | A100-80GB, x3.7 | A100-80GB, x7.5 |
|---|---|---|---|
| V3 cylinder, butterfly-fine 27968 x 64 modes, dt 0.002, 200 D/U (1e5 steps) | 66 h | 20 h | 10 h |
| V3 cylinder, 1e5-cell plane x 64 modes, same | ~240 h | ~70 h | ~35 h |
| channel Re_tau 395, 96x160 x 128 modes, 30 time units | 10 h | 3 h | 1.5 h |
| TGV Re 800/1600, 128^2 x 128, T 20 | 20 min | 6 min | 3 min |

Memory at V3 size (1e5 cells x 128 planes) is ~10 GB, so a 40 GB A100 is enough; more GPUs do
not help yet (single-device code).

## The runs

    # Taylor-Green Re 800 against the SEM DNS on disk (results/tgv_diag_re800_88.npz), section 58
    TGV_DEVICE=gpu TGV_RE=800 TGV_T=15 TGV_N=128 TGV_NZ=128 TGV_DT=0.02 TGV_SGS=wale python test_utgv3d.py A
    # channel Re_tau 180 against the FOSLS DNS window (results/fosls_chan180_stats_t5.2_30.npz), section 54
    python run_uchannel25.py --device gpu --T 30 --t-stats 10
    # spanwise-periodic cylinder (the V3 target at Re 3900 needs its own mesh; the Re 100 check is section 51)
    python run_ucylinder25.py meshes/cylinder_butterfly.msh --device gpu --nz 4 --T 150
    python run_ucylinder25.py meshes/cylinder_butterfly_fine.msh --device gpu --nz 64 --Lz 3.14159 --Re 3900 --dt 0.002 --sgs wale --T 200   # the V3 shape; needs its own Re 3900 mesh first

Both drivers take `--device gpu` (block AMG solves, momentum tolerance 1e-7) and were smoke-tested
on the GB10. Inside them every field is a CuPy array on the GPU path; the drivers read back through
`s.host(...)`, and `Stats` / `save` / `load` do the same.

## Memory

Per (ncell, nz) field 8 bytes x ncell x nz; the step holds ~40 such blocks plus the padded
(3nz/2) planes for the nonlinear term and the multigrid levels. A 1e5-cell plane x 128 planes
is ~10 GB at peak; the 80 GB A100 takes 4e5 cells x 128 planes.

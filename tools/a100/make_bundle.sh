#!/bin/bash
# Pack what the A100 run needs: the solver, the drivers, the meshes, the tools. Run from the repo root.
set -e
OUT=${1:-upict25_a100.tar.gz}
tar czf "$OUT" --exclude='__pycache__' \
  src/__init__.py src/umesh.py src/uops.py src/ugrad.py src/upiso.py src/upiso25.py src/umodesolve.py src/ucuda.py src/usgs.py src/ustats.py src/linsolve.py src/precond.py src/sgs.py \
  test_utgv3d.py run_uchannel25.py run_ucylinder25.py bench_modesolve.py test_uchannel_laminar.py test_usgs.py \
  meshes/cylinder_butterfly.msh meshes/cylinder_butterfly_fine.msh meshes/cylinder_butterfly_wake.msh meshes/channel_tri_graded.msh \
  results/minchan_re180_field.npz results/fosls_chan180_stats_t5.2_30.npz results/tgv_diag_re800_88.npz \
  tools/a100/
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"

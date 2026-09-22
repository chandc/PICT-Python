#!/bin/bash
# One-time setup on the Spark: persistent Firedrake container 'hgcmp' with HydroGym installed.
# Work dir ~/hydrogym_cmp is mounted at /work. Run: bash ~/hydrogym_cmp/spark_setup.sh
set -e
IMG=firedrakeproject/firedrake-vanilla-default:latest
docker rm -f hgcmp 2>/dev/null || true
docker run -d --name hgcmp -v ~/hydrogym_cmp:/work -w /work $IMG sleep infinity
echo "--- python / firedrake in image"
docker exec hgcmp bash -lc 'which python; python -c "import firedrake, sys; print(\"firedrake OK\", sys.version.split()[0])"; nproc'
echo "--- install hydrogym (PyPI 1.0.0) + firedrake extras"
docker exec hgcmp bash -lc 'pip install -q "hydrogym[firedrake]" 2>&1 | tail -3; python -c "import hydrogym, os; d=os.path.dirname(hydrogym.__file__); print(hydrogym.__version__ if hasattr(hydrogym,\"__version__\") else \"?\", d); print(sorted(os.listdir(d+\"/firedrake/envs/cylinder\")))"'
echo "--- import check"
docker exec hgcmp bash -lc 'python -c "import hydrogym.firedrake as hgym; f=hgym.Cylinder(Re=100, mesh=\"medium\", use_HF_data_manager=False); print(\"cells\", f.mesh.num_cells())"'

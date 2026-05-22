#!/bin/bash
#SBATCH --job-name=rfw-smoke
#SBATCH --output=smoke_%j.log
#SBATCH --error=smoke_%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=main
#
# Smoke tests for the Recurrent Foundation Wrapper. Runs on a compute node
# (we are on a login node — never invoke python directly).
#
# Usage from the login node:
#   sbatch run_smoke.sh                          # default: both tests
#   SMOKE_TEST=generic sbatch run_smoke.sh       # fast generic only (~2 min)
#   SMOKE_TEST=eqv2    sbatch run_smoke.sh       # Equiformer end-to-end only
#   SMOKE_TEST=param   sbatch run_smoke.sh       # print param summary
#
# After dispatch, wait for the job (squeue -u $USER), then read smoke_<jobid>.log.

source ~/miniconda3/bin/activate
conda activate obelix
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "=== RFW smoke ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Test selector: ${SMOKE_TEST:-all}"
echo ""

TEST=${SMOKE_TEST:-all}

if [[ "$TEST" == "all" || "$TEST" == "generic" ]]; then
    echo "--- rfw.test_smoke (generic, dummy model) ---"
    python -u -m rfw.test_smoke
    echo ""
fi

if [[ "$TEST" == "all" || "$TEST" == "eqv2" ]]; then
    echo "--- rfw_eqv2.test_smoke (real EquiformerV3 on CPU) ---"
    python -u -m rfw_eqv2.test_smoke
    echo ""
fi

if [[ "$TEST" == "param" ]]; then
    echo "--- Param summary ---"
    python -u -c "
from rfw.wrapper import RecurrentFoundationWrapper, RFWConfig
from rfw_eqv2.adapter import EqV2Adapter, _load_equiformer_v3
m = _load_equiformer_v3(device='cpu')
w = RecurrentFoundationWrapper(m, EqV2Adapter(m), feature_dim=m.num_channels, cfg=RFWConfig())
print(w.param_summary())
"
fi

echo "=== Done ==="
echo "Date: $(date)"

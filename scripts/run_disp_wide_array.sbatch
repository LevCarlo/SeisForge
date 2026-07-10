#!/usr/bin/env bash
#
# Run one wide dispersion inversion per Slurm array task.
# Submit from the parent of INV, or provide ROOT_DIR explicitly:
#   sbatch scripts/run_disp_wide_array.sbatch
#   sbatch --export=ALL,ROOT_DIR=/public2/home/user38/Guangxi.Data/INV.DATA/INV scripts/run_disp_wide_array.sbatch
#
# The 232 station directories are bc000 through bc231. At most 16 tasks run
# concurrently; each task receives six CPUs for its six concurrent chains.

#SBATCH --job-name=sf-wide-disp
#SBATCH --array=0-231%16
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --output=/public2/home/user38/Guangxi.Data/INV.DATA/slurm_%x_%A_%a.out
#SBATCH --error=/public2/home/user38/Guangxi.Data/INV.DATA/slurm_%x_%A_%a.err

# Add site-specific account, partition, or memory requests here when needed.
##SBATCH --partition=<partition>
##SBATCH --account=<account>
##SBATCH --mem=12G

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/public2/home/user38/Guangxi.Data/INV.DATA/INV}"
RUN_NAME="${RUN_NAME:-wide_v1}"
CONDA_HOME="${CONDA_HOME:-$HOME/software/miniforge3}"

# Each MCMC chain is already parallelized with a Python thread. Keep numerical
# kernels single-threaded so one array task remains within its six CPUs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LC_ALL=C

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate seis

shopt -s nullglob
stations=("${ROOT_DIR}"/bc[0-9][0-9][0-9])
if (( ${#stations[@]} != 232 )); then
    echo "Expected 232 bcNNN directories under ${ROOT_DIR}; found ${#stations[@]}." >&2
    exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#stations[@]} )); then
    echo "Invalid array index: ${SLURM_ARRAY_TASK_ID}." >&2
    exit 2
fi

station_dir="${stations[$SLURM_ARRAY_TASK_ID]}"
station_name="$(basename "${station_dir}")"
output_dir="${station_dir}/${RUN_NAME}"

if [[ ! -f "${station_dir}/inv.yaml" || ! -f "${station_dir}/obs.yaml" ]]; then
    echo "${station_name}: missing inv.yaml or obs.yaml." >&2
    exit 2
fi
if [[ -e "${output_dir}" ]]; then
    echo "${station_name}: output already exists: ${output_dir}" >&2
    echo "Refusing to overwrite an existing run. Use a new RUN_NAME or clean this station's failed output before retrying." >&2
    exit 3
fi

echo "[$(date -Is)] start ${station_name} on ${HOSTNAME}; CPUs=${SLURM_CPUS_PER_TASK}"
echo "conda: $(command -v python)"
echo "seisforge: $(command -v seisforge)"

cd "${station_dir}"
seisforge inv disp --inv inv.yaml --obs obs.yaml --output "${RUN_NAME}" --no-progress

touch "${output_dir}/.completed"
echo "[$(date -Is)] complete ${station_name}"

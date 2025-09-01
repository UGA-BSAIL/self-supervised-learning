set -e

# Base directory we use for job output.
OUTPUT_BASE_DIR="/blue/cli2/$(whoami)/job_scratch"
# Directory where our data and venv are located.
LARGE_FILES_DIR="/blue/cli2/$(whoami)/ssl/"
# Working directory for this job.
JOB_DIR="${OUTPUT_BASE_DIR}/job_${SLURM_JOB_ID}"

function copy_data_to_scratch() {
  # Copies primary data to a scratch directory on the local node for faster
  # loading.
  # Manually create links to the primary data.
  rm "${JOB_DIR}/data/05_model_input"
  mkdir "${JOB_DIR}/data/05_model_input"
  find "${LARGE_FILES_DIR}/data/05_model_input/"* -maxdepth 0 \
    -exec ln -s -t "${JOB_DIR}/data/05_model_input/" {} \;

  # Create output directories.
  if [ ! -d "${JOB_DIR}/output_data" ]; then
    mkdir "${JOB_DIR}/output_data"
  fi
  rm -rf "${JOB_DIR}/logs"
  mkdir -p "${JOB_DIR}/logs"

  # Copy the data.
  data_dir="${SLURM_TMPDIR}/data/"
  primary_dir="${JOB_DIR}/data/05_model_input"
  if [ ! -d "${data_dir}" ]; then
    mkdir "${data_dir}"

    echo "Extracting data..."
    find "${primary_dir}" -name "*.tar" -print0 | \
      xargs -0 -P 8 -I % tar -xf % --directory "${data_dir}"
  fi
  # Create symlinks.
  find "${data_dir}/"* -maxdepth 0 -exec ln -fs -t "${primary_dir}/" {} \;
}

function prepare_environment() {
  # Create the working directory for this job.
  mkdir "${JOB_DIR}"
  echo "Job directory is ${JOB_DIR}."

  # Copy the code.
  cp -Rd "${SLURM_SUBMIT_DIR}/"* "${JOB_DIR}/"

  # Create the venv.
  ml python/3.10
  if [ ! -d "${SLURM_TMPDIR}/venv" ]; then
    python -m venv "${SLURM_TMPDIR}/venv"
  fi
  rm -f .venv
  ln -s "${SLURM_TMPDIR}/venv" .venv
  source .venv/bin/activate
  poetry install --only main --no-root

  # Link to the input data directory.
  rm -rf "${JOB_DIR}/data"
  mkdir "${JOB_DIR}/data"
  find "${LARGE_FILES_DIR}/data/"* -maxdepth 0 \
    -exec ln -s -t "${JOB_DIR}/data/" {} \;

  # Set the working directory correctly for Kedro.
  cd "${JOB_DIR}"
  # Remove temporary files.
  rm -rf wandb
}

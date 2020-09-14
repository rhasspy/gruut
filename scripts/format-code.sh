#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

venv="${src_dir}/.venv"
if [[ -d "${venv}" ]]; then
    echo "Using virtual environment at ${venv}"
    source "${venv}/bin/activate"
fi

python_files=("${src_dir}/gruut/"*.py)
python_files+=("${src_dir}/tests/"*.py)

# -----------------------------------------------------------------------------

black "${python_files[@]}"
isort "${python_files[@]}"

# -----------------------------------------------------------------------------

echo "OK"

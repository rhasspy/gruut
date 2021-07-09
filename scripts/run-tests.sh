#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

venv="${src_dir}/.venv"
if [[ -d "${venv}" ]]; then
    source "${venv}/bin/activate"
fi

# -----------------------------------------------------------------------------

export PYTHONPATH="${src_dir}"

coverage run "--source=${src_dir}/gruut" -m pytest
coverage report -m
coverage xml

# -----------------------------------------------------------------------------

echo "OK"

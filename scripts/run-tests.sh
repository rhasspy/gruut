#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

if [[ "$1" == '--no-venv' ]]; then
    no_venv='1'
fi

if [[ -z "${no_venv}" ]]; then
    venv="${src_dir}/.venv"
    if [[ -d "${venv}" ]]; then
        source "${venv}/bin/activate"
    fi
fi

# -----------------------------------------------------------------------------

export PYTHONPATH="${src_dir}"

while read -r lang_dir;
do
    export PYTHONPATH="${lang_dir}:${PYTHONPATH}"
done < <(find "${src_dir}" -maxdepth 1 -type d -name 'gruut-lang-*')

coverage run "--source=${src_dir}/gruut" -m pytest
coverage report -m
coverage xml

# -----------------------------------------------------------------------------

echo "OK"

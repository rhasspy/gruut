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

python_files=("${src_dir}/gruut/"*.py "${src_dir}/bin/"*.py)

# Add language data modules
while read -r lang_module_dir; do
    if [[ -f "${lang_module_dir}/__init__.py" ]]; then
        lang_dir="$(dirname "${lang_module_dir}")"
        python_files+=("${lang_module_dir}/"*.py "${lang_dir}/"*.py)
    fi
done < <(find "${src_dir}" -mindepth 2 -maxdepth 2 -name 'gruut_lang_*' -type d)

# -----------------------------------------------------------------------------

black "${python_files[@]}"
isort "${python_files[@]}"

# -----------------------------------------------------------------------------

echo "OK"

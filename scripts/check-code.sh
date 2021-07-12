#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

venv="${src_dir}/.venv"
if [[ -d "${venv}" ]]; then
    source "${venv}/bin/activate"
fi

python_files=("${src_dir}/gruut/"*.py)

# -----------------------------------------------------------------------------

function check_code {
    flake8 "$@"
    pylint "--rcfile=${src_dir}/.pylintrc" "$@"
    mypy "$@"
    black --check "$@"
    isort --check-only "$@"
}

echo "${src_dir}/gruut"
check_code "${python_files[@]}"

# Add language data modules
while read -r lang_module_dir; do
    if [[ -f "${lang_module_dir}/__init__.py" ]]; then
        lang_dir="$(dirname "${lang_module_dir}")"
        python_files=("${lang_module_dir}/"*.py "${lang_dir}/"*.py)

        echo ""
        echo "${lang_dir}"
        check_code "${python_files[@]}"
    fi
done < <(find "${src_dir}" -mindepth 2 -maxdepth 2 -name 'gruut_lang_*' -type d)

# -----------------------------------------------------------------------------

echo "OK"

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

python_files=("${src_dir}/gruut/"*.py "${src_dir}/tests/"*.py "${src_dir}/setup.py")

# Add bin scripts selectively
bin_scripts=('clean-metadata' 'fst2npy' 'map_lexicon' 'phonemize_lexicon' 'reorder_lexicon' 'espeak_word')

while read -r python_lib; do
    if [ "$(echo "${python_lib}" | grep 'phonetisaurus')" ]; then
        bin_scripts+=('phonetisaurus_per')
    elif [ "$(echo "${python_lib}" | grep 'aeneas')" ]; then
        bin_scripts+=('librivox_align')
    fi
done < <(pip3 freeze)

for script_name in "${bin_scripts[@]}"; do
    python_files+=("${src_dir}/bin/${script_name}.py")
done

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

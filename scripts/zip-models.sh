#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

dist_dir="${src_dir}/dist"
mkdir -p "${dist_dir}"

find "${src_dir}" -mindepth 1 -maxdepth 1 -name 'gruut_lang_*' -type d | \
    while read -r lang_dir;
    do
        if [[ ! -f "${lang_dir}/setup.py" ]]; then
            # Skip
            continue
        fi

        pushd "${lang_dir}" > /dev/null
        rm -rf dist/ "*.egg-info/"
        python3 setup.py sdist
        mv dist/* "${dist_dir}/"
        rm -rf dist/
        popd > /dev/null

        echo "${lang_dir}"
    done

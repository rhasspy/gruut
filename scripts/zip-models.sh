#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

dist_dir="${src_dir}/dist"
mkdir -p "${dist_dir}/data"

rm -rf -- *.egg-info/

find "${src_dir}" -mindepth 1 -maxdepth 1 -name 'gruut-lang-*' -type d | \
    while read -r lang_dir;
    do
        if [[ ! -f "${lang_dir}/setup.py" ]]; then
            # Skip
            continue
        fi

        pushd "${lang_dir}" > /dev/null

        # Create Python package distribution
        rm -rf dist/ -- *.egg-info/
        python3 setup.py sdist
        mv dist/* "${dist_dir}/"
        rm -rf dist/

        # Create standalone distribution
        lang_dir_name="$(basename "${lang_dir}" | sed 's/-/_/g')"
        full_lang="$(awk '{print $1}' LANGUAGE)"
        tar -czf "${dist_dir}/data/${full_lang}.tar.gz" \
            --transform "s/${lang_dir_name}/${full_lang}/" "${lang_dir_name}"

        popd > /dev/null

        echo "${lang_dir}"
    done

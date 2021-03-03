#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

dist_dir="${src_dir}/dist/data"
mkdir -p "${dist_dir}"

data_dir="${src_dir}/data"
find "${data_dir}" -mindepth 1 -maxdepth 1 -type d | \
    while read -r lang_dir;
    do
        if [[ ! -f "${lang_dir}/language.yml" ]]; then
            # Skip incomplete languages
            continue
        fi

        lang="$(basename "${lang_dir}")"
        lang_file="${dist_dir}/${lang}.tar.gz"

        rm -f "${lang_file}"
        pushd "${lang_dir}" > /dev/null
        tar -czf "${lang_file}" --exclude=lexicon.txt --exclude=g2p.corpus *
        popd > /dev/null

        echo "${lang_file}"
    done

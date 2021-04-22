#!/usr/bin/env bash
temp_dir="$(mktemp -d)"

function cleanup {
    rm -rf "${temp_dir}"
}

trap cleanup EXIT

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

gruut="${src_dir}/bin/gruut"
test_sentences="${src_dir}/test/test_sentences.txt"

while read -r line; do
    if [ -z "${line}" ] || [ "${line:0:1}" == '#' ]; then
        # Skip blank lines and comments
        continue
    fi

    IFS='|' read -r lang text truth < <(echo "${line}")

    "${gruut}" "${lang}" tokenize "${text}" | \
        "${gruut}" "${lang}" phonemize --word-breaks | \
        jq --raw-output '.pronunciation_text' \
           > "${temp_dir}/guess.txt"

    diff --side-by-side --width=50 "${temp_dir}/guess.txt" <(echo "${truth}") > "${temp_dir}/diff.txt"
    if [[ $? -ne 0 ]]; then
            echo "${lang}: <(cat "${temp_dir}/diff.txt")"
            fail='1'
    fi
done < "${test_sentences}"

if [[ -n "${fail}" ]]; then
    echo 'Tests failed'
    exit 1;
fi

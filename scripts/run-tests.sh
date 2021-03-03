#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

gruut="${src_dir}/bin/gruut"
test_sentences="${src_dir}/test/test_sentences.txt"

while read -r lang text; do
    "${gruut}" "${lang}" tokenize "${text}" | \
        "${gruut}" "${lang}" phonemize | \
        jq -r '.pronunciation_text' | \
        xargs -0 printf '%s %s' "${lang}"
done < "${test_sentences}"

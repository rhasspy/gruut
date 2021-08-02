#!/usr/bin/env bash
set -e

if [[ -z "$1" ]]; then
    echo "Usage: run-g2p-tests.sh <LANG> [--pos]"
    exit 0
fi

lang="$1"

if [ "${lang}" == 'en-us' ] || [ "$2" == '--pos' ]; then
    pos='1'
fi

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"
bin_dir="${src_dir}/bin"

venv="${src_dir}/.venv"
if [[ -d "${venv}" ]]; then
    source "${venv}/bin/activate"
fi

export PYTHONPATH="${src_dir}"

# -----------------------------------------------------------------------------

function test_g2p {
    lang_dir="$1"

    lexicon="${lang_dir}/lexicon.txt"

    if [[ -n "${pos}" ]]; then
        # Create version of lexicon without part of speech tags
        lexicon_nopos="${lang_dir}/lexicon.nopos.txt"
        if [[ ! -s "${lexicon_nopos}" ]]; then
            cut -d' ' -f1,3- < "${lexicon}" > "${lexicon_nopos}"
        fi

        lexicon="${lexicon_nopos}"
    fi

    # Generate test lexicons
    lexicon10k="${lang_dir}/lexicon.10k.txt"
    lexicon100="${lang_dir}/lexicon.100.txt"

    if [[ ! -s "${lexicon10k}" ]]; then
        shuf -n 10000 < "${lexicon}" > "${lexicon10k}"
    fi

    if [[ ! -s "${lexicon100}" ]]; then
        shuf -n 100 < "${lexicon}" > "${lexicon100}"
    fi

    g2p_results="${lang_dir}/g2p_results.txt"

    # CRF
    if [ ! -s "${g2p_results}" ] || ! grep -q '^CRF:' "${g2p_results}"; then
        g2p_crf="${lang_dir}/g2p/model.crf"
        echo "Testing ${g2p_crf}"

        python3 -m gruut.g2p test \
                --model "${g2p_crf}" \
                < "${lexicon10k}" | \
            awk '{print "CRF: " $0}' | \
            tee --append "${g2p_results}"

        echo ''
    fi

    # FST
    if [ ! -s "${g2p_results}" ] || ! grep -q '^FST:' "${g2p_results}"; then
        g2p_fst="${lang_dir}/g2p.fst"
        echo "Testing ${g2p_fst}"

        python3 "${bin_dir}/phonetisaurus_per.py" \
                "${g2p_fst}" \
                < "${lexicon10k}" | \
            awk '{print "FST: " $0}' | \
            tee --append "${g2p_results}"

        echo ''
    fi

    # NPZ
    if [ ! -s "${g2p_results}" ] || ! grep -q '^NPZ:' "${g2p_results}"; then
        g2p_npz="${lang_dir}/g2p/graph.npz"
        echo "Testing ${g2p_npz}"

        python3 -m gruut.g2p_phonetisaurus test \
                --graph "${g2p_npz}" --preload \
                < "${lexicon100}" | \
            awk '{print "NPZ: " $0}' | \
            tee --append "${g2p_results}"

        echo ''
    fi
}

# -----------------------------------------------------------------------------

lang_dir="${src_dir}/data/${lang}"

test_g2p "${lang_dir}"
test_g2p "${lang_dir}/espeak"


echo 'Done'

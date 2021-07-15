#!/usr/bin/env bash
set -e

if [[ -z "$1" ]]; then
    echo "Usage: build_espeak_db.sh <LANG>"
    exit 0
fi

lang="$1"

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

espeak_dir="${src_dir}/data/${lang}/espeak"
mkdir -p "${espeak_dir}"

if [[ "${lang}" == 'en-us' ]]; then
    pos='1'
fi

# Text lexicon
lexicon_text="${espeak_dir}/lexicon.txt"
lexicon_args=()

if [[ ! -s "${lexicon_text}" ]]; then
    words="${espeak_dir}/words.txt"
    if [[ ! -f "${words}" ]]; then
        echo "Missing ${words}"
        exit 1
    fi

    echo "Creating lexicon (${lexicon_text})"

    if [[ -n "${pos}" ]]; then
        # Use part of speech with English
        lexicon_args+=('--pos')
    fi

    python3 "${bin_dir}/espeak_word.py" "${lang}" "${lexicon_args[@]}" \
            < "${words}" > "${lexicon_text}"
fi

# Database lexicon
lexicon_db="${espeak_dir}/lexicon.db"
echo "Adding lexicon to database (${lexicon_db})"
python3 -m gruut.lexicon2db \
        --casing lower \
        --lexicon "${lexicon_text}" \
        --database "${lexicon_db}" "${lexicon_args[@]}"

# Create g2p model
g2p_corpus="${espeak_dir}/g2p.corpus"
g2p_fst="${espeak_dir}/g2p.fst"

if [ ! -s "${g2p_corpus}" ] || [ ! -s "${g2p_fst}" ]; then
    echo "Creating Phonetisaurus g2p model (${g2p_fst})"

    lexicon_g2p="${lexicon_text}"
    if [[ -n "${pos}" ]]; then
        # Drop POS column from lexicon
        lexicon_g2p="$(basename "${lexicon_text}" .txt).nopos.txt"
        cut -d' ' -f1,3- < "${lexicon_text}" > "${lexicon_g2p}"
    fi

    phonetisaurus train \
                  --corpus "${g2p_corpus}" \
                  --model "${g2p_fst}" \
                  "${lexicon_g2p}"
fi

g2p_dir="${espeak_dir}/g2p"
mkdir -p "${g2p_dir}"

# Convert Phonetisaurus graph
graph_npz="${g2p_dir}/graph.npz"
python3 "${bin_dir}/fst2npy.py" \
        <(fstprint "${g2p_fst}") \
        "${graph_npz}"

# Add g2p alignments to database
echo "Adding g2p alignments to database (${lexicon_db})"
python3 -m gruut.corpus2db \
        --corpus "${g2p_corpus}" \
        --database "${lexicon_db}"

# Train CRF model
g2p_crf="${g2p_dir}/model.crf"
if [[ ! -s "${g2p_crf}" ]]; then
    echo "Training g2p crf model (${g2p_crf})"
    python3 -m gruut.g2p train \
            --corpus "${g2p_corpus}" \
            --output "${g2p_crf}"
fi

echo 'Done'

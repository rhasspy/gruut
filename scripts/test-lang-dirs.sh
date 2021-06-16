#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

venv="${src_dir}/.venv"
if [[ -d "${venv}" ]]; then
    source "${venv}/bin/activate"
fi

export PYTHONPATH="${src_dir}"

# -----------------------------------------------------------------------------

function gruut {
    lang="$1"
    text="$2"
    shift 2

    python3 -m gruut "${lang}" tokenize "${text}" | \
        python3 -m gruut "${lang}" phonemize "$@" | \
        jq -r .pronunciation_text
}

# -----------------------------------------------------------------------------

declare -A sentences
sentences['cs-cz']="Jeden jazyk nikdy nestačí."
sentences['de-de']="Eine Sprache ist niemals genug."
sentences['en-us']="One language is never enough."
sentences['es-es']="Un idioma nunca es suficiente."
sentences['fr-fr']="Une seule langue n'est jamais suffisante."
sentences['it-it']="Una sola lingua non è mai abbastanza."
sentences['nl']="Een enkele taal is nooit genoeg."
sentences['pt']="Uma só língua nunca basta."
sentences['ru-ru']="Одного языка никогда недостаточно."
sentences['sv-se']="Ett språk är aldrig nog."
sentences['sw']="Lugha moja haitoshi."

# English
full_lang='en-us'
sentence="${sentences["${full_lang}"]}"

# With gruut phonemes
phonemes="$(gruut "${full_lang}" "${sentence}")"
echo "${full_lang}: ${phonemes}"

# With espeak phonemes
espeak_phonemes="$(gruut "${full_lang}" "${sentence}" --model-prefix espeak)"
echo "${full_lang}: ${espeak_phonemes}"


find "${src_dir}" -mindepth 1 -maxdepth 1 -name 'gruut-lang-*' -type d | \
    while read -r lang_dir; do
          if [[ ! -f "${lang_dir}/setup.py" ]]; then
              # Skip
              continue
          fi

          full_lang="$(cat "${lang_dir}/LANGUAGE")"
          sentence="${sentences["${full_lang}"]}"

          # With gruut phonemes
          phonemes="$(gruut "${full_lang}" "${sentence}")"
          echo "${full_lang}: ${phonemes}"

          # With espeak phonemes
          espeak_phonemes="$(gruut "${full_lang}" "${sentence}" --model-prefix espeak)"
          echo "${full_lang}: ${espeak_phonemes}"
    done

# -----------------------------------------------------------------------------

echo "OK"

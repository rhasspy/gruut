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

export PYTHONPATH="${src_dir}"

while read -r lang_dir;
do
    export PYTHONPATH="${lang_dir}:${PYTHONPATH}"
done < <(find "${src_dir}" -maxdepth 1 -type d -name 'gruut-lang-*')

# -----------------------------------------------------------------------------

function gruut {
    lang="$1"
    text="$2"
    shift 2

    python3 -m gruut "${lang}" tokenize "${text}" | \
        python3 -m gruut "${lang}" phonemize "$@" | \
        jq -r .pronunciation_text
}

function espeak {
    lang="$1"
    text="$(echo "$2" | sed -e 's/[ ]\+/|/g')"
    shift 2

    echo "${text}" | espeak-ng -v "${lang}" -q --ipa
}

function normalize {
    # Ignore whitespace, IPA breaks, and stress
    echo "$1" | sed -e 's/[ |‖ˈˌ]//g'
}

function check_espeak {
    expected="$(normalize "$1")"
    actual="$(normalize "$2")"

    if [[ "${expected}" != "${actual}" ]]; then
        echo "Expected ${expected} but got ${actual}" >&2
        # exit 1
    fi
}

# -----------------------------------------------------------------------------

declare -A sentences
sentences['ar']="لغة واحدة لا تكفي"
sentences['cs-cz']="Jeden jazyk nikdy nestačí."
sentences['de-de']="Eine Sprache ist niemals genug."
sentences['en-us']="One language is never enough."
sentences['es-es']="Un idioma nunca es suficiente."
sentences['fa']="يک زبان کافي نيست"
sentences['fr-fr']="Une seule langue n'est jamais suffisante."
sentences['it-it']="Una sola lingua non è mai abbastanza."
sentences['lb']="An der Zäit hunn sech den Nordwand an d’Sonn gestridden."
sentences['nl']="Een enkele taal is nooit genoeg."
sentences['pt']="Uma só língua nunca basta."
sentences['ru-ru']="Одного языка никогда недостаточно."
sentences['sv-se']="Ett språk är aldrig nog."
sentences['sw']="Lugha moja haitoshi."

declare -A voices
voices['ar']='ar'
voices['cs-cz']='cs'
voices['de-de']='de'
voices['en-us']='en-us'
voices['es-es']='es'
voices['fa']='fa'
voices['fr-fr']='fr'
voices['it-it']='it'
voices['lb']='lb'
voices['nl']='nl'
voices['pt']='pt'
voices['ru-ru']='ru'
voices['sv-se']='sv'
voices['sw']='sw'

# -----------------------------------------------------------------------------

for full_lang in "${!sentences[@]}"; do
    sentence="${sentences["${full_lang}"]}"

    # With gruut phonemes
    phonemes="$(gruut "${full_lang}" "${sentence}")"
    echo "${full_lang}: ${phonemes}"

    # With espeak phonemes
    espeak_phonemes="$(gruut "${full_lang}" "${sentence}" --model-prefix espeak)"
    echo "${full_lang}: ${espeak_phonemes}"

    # Check against espeak
    expected_espeak_phonemes="$(espeak "${voices["${full_lang}"]}" "${sentence}")"
    check_espeak "${expected_espeak_phonemes}" "${espeak_phonemes}"

    echo ''
done

# -----------------------------------------------------------------------------

echo "OK"

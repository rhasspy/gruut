#!/usr/bin/env bash
if [[ -z "$1" ]]; then
    echo 'Usage: espeak_word.sh VOICE [pos] < words > lexicon'
    exit 1
fi

voice="$1"
pos="$2"

function get_ipa {
    voice="$1"
    text="$2"
    espeak-ng -q --ipa --sep=' ' -v "${voice}" "${text}" | sed -e 's/^[ ]\+//'
}

function drop_word {
    voice="$1"
    text="$2"
    espeak-ng -q --ipa --sep='|' -v "${voice}" "${text}" | sed -e 's/^[ ]\+//' | cut -d' ' -f2- | sed s'/|/ /g'
}

while read -r word;
do
    ipa="$(get_ipa "${voice}" "${word}")"

    if [[ -z "${pos}" ]]; then
        echo "${word} ${ipa}"
    else
        # Part of speech
        verb="$(drop_word "${voice}" "preferably ${word}")"
        noun="$(drop_word "${voice}" "a ${word}")"
        past="$(drop_word "${voice}" "had ${word}")"

        # Default form
        echo "${word} _ ${ipa}"

        if [[ "${ipa}" != "${verb}" ]]; then
            # Verb form
            echo "${word} VB ${verb}"
        fi

        if [[ "${ipa}" != "${noun}" ]]; then
            # Noun form
            echo "${word} NN ${noun}"
        fi

        if [[ "${ipa}" != "${past}" ]]; then
            # Past tense verb form
            echo "${word} VBD ${past}"
        fi
    fi
done


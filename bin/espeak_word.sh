#!/usr/bin/env bash
if [[ -z "$1" ]]; then
    echo 'Usage: espeak_word.sh VOICE < words > lexicon'
    exit 1
fi

voice="$1"

while read -r word;
do
    ipa="$(espeak-ng -q --ipa --sep=' ' -v "${voice}" "${word}" | sed -e 's/^[ ]\+//')"
    echo "${word} ${ipa}"
done


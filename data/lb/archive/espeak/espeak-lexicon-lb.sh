filename="/home/mbarnig/Downloads/lb-lb/LOD/LOD-words.txt"
lexiconfile="/home/mbarnig/Downloads/lb-lb/LOD/lexicon-LOD-espeak-2.txt"

function get_ipa {
    text="$1"
    espeak-ng -q --ipa --sep=' ' -v "lb-lb" "${text}" | sed -e 's/^[ ]\+//'
}

while read word; do
    ipa="$(get_ipa "${word}")"
    echo "${word} ${ipa}" >> $lexiconfile
done < $filename

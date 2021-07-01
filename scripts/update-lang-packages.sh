#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

# -----------------------------------------------------------------------------

# Update setup.py, VERSION, and language files files
echo "Language data modules"
find "${src_dir}" -mindepth 1 -maxdepth 1 -name 'gruut-lang-*' -type d | \
    while read -r lang_dir; do
        # Update setup.py
        full_lang="$(cat "${lang_dir}/LANGUAGE")"
        IFS='-' read -ra lang_parts <<< "${full_lang}"
        lang_code="${lang_parts[0]}"

        LANG_NAME="${full_lang}" LANG_CODE="${lang_code}" \
                 envsubst < "${src_dir}/etc/lang_setup.py.in" \
                 > "${lang_dir}/setup.py"

        module_dir="${lang_dir}/gruut_lang_${lang_code}"

        # Update version
        cp "${src_dir}/gruut/VERSION" \
           "${module_dir}/"

        # Update lexicon.db and g2p/model.crf
        data_dir="${src_dir}/data/${full_lang}"

        cp "${data_dir}/lexicon.db" \
           "${module_dir}/"

        cp "${data_dir}/g2p/model.crf" \
           "${module_dir}/g2p/"

        cp "${data_dir}/espeak/lexicon.db" \
           "${module_dir}/espeak/"

        cp "${data_dir}/espeak/g2p/model.crf" \
           "${module_dir}/espeak/g2p/"

        echo "${full_lang} ${lang_code} ${lang_dir}"
    done

# -----------------------------------------------------------------------------

echo ""
echo "g2p data modules"

# -----------------------------------------------------------------------------

echo ""
echo "OK"

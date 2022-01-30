#!/usr/bin/env bash
set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

# -----------------------------------------------------------------------------

lang_version='2.0.1'

# Update setup.py, VERSION, and language files files
echo "Language data modules"
find "${src_dir}" -mindepth 1 -maxdepth 1 -name 'gruut-lang-*' -type d | \
    while read -r lang_dir; do
        # Update setup.py
        full_lang="$(awk '{print $1}' "${lang_dir}/LANGUAGE")"
        IFS='-' read -ra lang_parts <<< "${full_lang}"
        lang_code="${lang_parts[0]}"
        lang_name="$(awk '{print $2}' "${lang_dir}/LANGUAGE")"

        LANG_NAME="${lang_name}" LANG_CODE="${lang_code}" \
                 envsubst < "${src_dir}/etc/lang_setup.py.in" \
                 > "${lang_dir}/setup.py"

        LANG_NAME="${lang_name}" LANG_CODE="${lang_code}" \
                 envsubst < "${src_dir}/etc/lang_README.md.in" \
                 > "${lang_dir}/README.md"

        module_dir="${lang_dir}/gruut_lang_${lang_code}"

        LANG_NAME="${lang_name}" LANG_CODE="${lang_code}" \
                 envsubst < "${src_dir}/etc/__init__.py.in" \
                 > "${module_dir}/__init__.py"

        # Update version
        if [[ ! -s "${module_dir}/VERSION" ]]; then
            echo "${lang_version}" > "${module_dir}/VERSION";
        fi

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

        if [[ -d "${data_dir}/pos" ]]; then
            mkdir -p "${module_dir}/pos"
            if [[ -f "${data_dir}/pos/model.crf" ]]; then
                cp "${data_dir}/pos/model.crf" \
                   "${module_dir}/pos/"
            elif [[ -f "${data_dir}/pos/postagger.model" ]]; then
                cp "${data_dir}/pos/postagger.model" \
                   "${module_dir}/pos/"
            else
                echo "No POS model in ${data_dir}/pos"
                exit 1
            fi
        fi

        echo "${full_lang} ${lang_code} ${lang_dir}"
    done

# -----------------------------------------------------------------------------

echo ""
echo "OK"

#!/usr/bin/env bash
set -e

if [[ -z "${PIP_INSTALL}" ]]; then
    PIP_INSTALL='install'
fi

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

# -----------------------------------------------------------------------------

venv="${src_dir}/.venv"

# -----------------------------------------------------------------------------

function maybe_download {
    if [[ ! -s "$2" ]]; then
        if [[ -n "${offline}" ]]; then
            echo "Need to download $1 but offline."
            exit 1
        fi

        mkdir -p "$(dirname "$2")"
        curl -sSfL -o "$2" "$1" || { echo "Can't download $1"; exit 1; }
        echo "$1 => $2"
    fi
}

# spaCy models
en_url='https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz'
en_file='en_core_web_sm-2.3.0.tar.gz'
maybe_download "${en_url}" "${en_file}"

nl_url='https://github.com/explosion/spacy-models/releases/download/nl_core_news_sm-2.3.0/nl_core_news_sm-2.3.0.tar.gz'
nl_file="${download}/nl_core_news_sm-2.3.0.tar.gz"
maybe_download "${nl_url}" "${nl_file}"

# -----------------------------------------------------------------------------

: "${PYTHON=python3}"

python_version="$(${PYTHON} --version)"

# Create virtual environment
echo "Creating virtual environment at ${venv} (${python_version})"
rm -rf "${venv}"
"${PYTHON}" -m venv "${venv}"
source "${venv}/bin/activate"

# Install Python dependencies
echo "Installing Python dependencies"
pip3 ${PIP_INSTALL} --upgrade pip
pip3 ${PIP_INSTALL} --upgrade wheel setuptools

if [[ -f requirements.txt ]]; then
    pip3 ${PIP_INSTALL} -r requirements.txt
fi

# Install models
pip3 ${PIP_INSTALL} "${en_file}"
python3 -m spacy link 'en_core_web_sm' 'en-us'

pip3 ${PIP_INSTALL} "${nl_file}"
python3 -m spacy link 'nl_core_news_sm' 'nl'

# Development dependencies
if [[ -f requirements_dev.txt ]]; then
    pip3 ${PIP_INSTALL} -r requirements_dev.txt || echo "Failed to install development dependencies" >&2
fi

# -----------------------------------------------------------------------------

echo "OK"

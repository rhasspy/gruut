SHELL := bash

.PHONY: check clean reformat dist venv test docs

all: dist

check:
	scripts/check-code.sh

reformat:
	scripts/format-code.sh

venv:
	scripts/create-venv.sh

dist:
	python3 setup.py sdist
	scripts/zip-models.sh

test:
	scripts/run-tests.sh

docs:
	sphinx-apidoc -f -o docs/source gruut
	sphinx-build -b html docs/source/ docs/

SHELL := bash

.PHONY: check clean reformat dist venv

all: dist

check:
	scripts/check-code.sh

reformat:
	scripts/format-code.sh

venv:
	scripts/create-venv.sh

dist:
	python3 setup.py sdist

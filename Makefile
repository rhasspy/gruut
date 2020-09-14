SHELL := bash

.PHONY: check clean reformat dist

all: dist

check:
	scripts/check-code.sh

reformat:
	scripts/format-code.sh

dist:
	rm -rf build
	PLATFORM=x86_64 python3 setup.py bdist_wheel --plat-name=manylinux1_x86_64
	rm -rf build
	PLATFORM=armv6l python3 setup.py bdist_wheel --plat-name=linux_armv6l
	rm -rf build
	PLATFORM=armv7l python3 setup.py bdist_wheel --plat-name=linux_armv7l
	rm -rf build
	PLATFORM=armv8 python3 setup.py bdist_wheel --plat-name=linux_armv8

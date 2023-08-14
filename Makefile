# Copyright (c) 2023 tc-haung
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

.PHONY: help bash yapf notebook test

VENV_NAME?=venv
VENV_ACTIVATE=. $(VENV_NAME)/bin/activate
PYTHON=python
# PYTHON=${VENV_NAME}/bin/python3
IMAGE=robotic-arm-cpu:latest
SRC_DIR=.
NOTEBOOK_TOKEN=robotic-arm

.DEFAULT: help
help:
	@echo "make prepare-dev"
	@echo "       prepare development environment, use only once"
    # @echo "make test"
    # @echo "       run tests"
    # @echo "make lint"
    # @echo "       run pylint and mypy"
    # @echo "make run"
    # @echo "       run project"
    # @echo "make clean"
    # @echo "       clean python cache files"
    # @echo "make doc"
    # @echo "       build sphinx documentation"
bash:
	docker run --rm -it -v ./src:/tf/src -v ./tests:/tf/tests ${IMAGE} bash

notebook:
	docker run --rm -it -v ./src:/tf/src -v ./tests:/tf/tests -p 8888:8888 ${IMAGE}

yapf:
#	yapf --in-place --recursive --style='{based_on_style: google, indent_width: 4}' --parallel --print-modified --verbose ${SRC_DIR} 2>&1 >/dev/null
	yapf --in-place --recursive --style='{based_on_style: google, indent_width: 4}'  --parallel --print-modified --print-modified --verbose ${SRC_DIR}

# test:
#     python -m pytest

# lint:
#     ${PYTHON} -m pylint
#     ${PYTHON} -m mypy
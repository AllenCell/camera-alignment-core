# See https://tech.davis-hansson.com/p/make/
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

##############################################

PYTHON_VERSION = python3.9
VENV_NAME := venv
ACTIVATE = $(VENV_NAME)/bin/activate
PYTHON = $(VENV_NAME)/bin/python3

$(PYTHON):
> test -d $(VENV_NAME) || $(PYTHON_VERSION) -m venv $(VENV_NAME)
> $(PYTHON) -m pip install -U pip

venv: $(PYTHON)

install: venv requirements.txt setup.py
> $(PYTHON) -m pip install -r requirements.txt

lint: venv
> $(PYTHON) -m flake8 --count --exit-zero camera_alignment_core
.PHONY: lint

type-check: venv
> $(PYTHON) -m mypy --ignore-missing-imports camera_alignment_core
.PHONY: type-check

fmt:
> $(PYTHON) -m black camera_alignment_core
.PHONY: fmt

test: venv
> $(PYTHON) -m pytest
.PHONY: test

clean:  ## clean all generated files
> git clean -Xfd
.PHONY: clean

build: ## run tox / run tests and lint
> $(PYTHON) -m tox
.PHONY: build

docs:
> source $(ACTIVATE) && sphinx-apidoc -f -o docs camera_alignment_core camera_alignment_core/tests
> source $(ACTIVATE) && sphinx-build -b html docs docs/build
.PHONY: docs

docs-serve:
> $(PYTHON) -m http.server --directory docs/build 8080
.PHONY: docs-serve

bumpversion-release: venv
> $(PYTHON) -m bumpversion --list release
.PHONY: bumpversion-release

bumpversion-major: venv
> $(PYTHON) -m bumpversion --list major
.PHONY: bumpversion-major

bumpversion-minor: venv
> $(PYTHON) -m bumpversion --list minor
.PHONY: bumpversion-minor

bumpversion-patch: venv
> $(PYTHON) -m bumpversion --list patch
.PHONY: bumpversion-patch

bumpversion-dev: venv
> $(PYTHON) -m bumpversion --list devbuild
.PHONY: bumpversion-dev

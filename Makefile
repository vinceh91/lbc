# Makefile
SHELL := /bin/bash
.SHELLFLAGS = -ec
.ONESHELL:
.SILENT:

PROJECT_NAME?=priceless

.PHONY: help
help:
	echo "❓ Use \`make <target>' where <target> could be"
	grep -E '^\.PHONY: [a-zA-Z0-9_-]+ .*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = "(: |##)"}; {printf "\033[36m%-30s\033[0m %s\n", $$2, $$3}'

.PHONY: venv ## 🐍 creates development environment
venv:
	python -m venv venv
	source venv/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel && \
	python -m pip install -e ".[dev]" --no-cache-dir && \
	pre-commit install && \
	pre-commit autoupdate

.PHONY: dependencies ## ⏬  installs production dependencies
dependencies:
	python -m pip install -r requirements.txt

.PHONY: style ## 🏄‍refactors code app
style:
	black ${PROJECT_NAME}
	flake8 ${PROJECT_NAME}
	isort ${PROJECT_NAME}

.PHONY: dead-code ## ☠️ removes dead code
dead-code:
	vulture ${PROJECT_NAME}

.PHONY: static-type ## ✅  checks static types
static-type:
	python -m mypy --ignore-missing-imports ${PROJECT_NAME}

.PHONY: clear-notebook-outputs ## ❇️ clear outputs of notebooks
clear-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*

.PHONY: clean ## 🧹 cleans all files in package app
clean: dead-code static-type style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

.PHONY: update-package ## 📦 updates and packages app
update-package:
	pip uninstall ${PROJECT_NAME} -y
	python -m pip install -e ".[dev]"

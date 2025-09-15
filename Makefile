SHELL := /bin/bash
.ONESHELL:

.PHONY: create activate remove install exec/demo

CONDA_NAME = ParaDiffusion
PYTHON_VERSION = 3.10

create:
	conda create -y -n $(CONDA_NAME) python=$(PYTHON_VERSION) \
	&& conda activate $(CONDA_NAME) \
	&& pip install --upgrade pip \
	&& pip install -r requirements.txt.v2

install:
	pip install -r requirements.txt.v2

# eval "$(make activate)" で環境を有効化可能
activate:
	source activate base && \
	conda activate $(CONDA_NAME)

remove:
	conda env remove -y -n $(CONDA_NAME)

exec/demo:
	python demo.py

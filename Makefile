.PHONY: setup ingest build train test app all clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e '.[dev]'

ingest:
	. .venv/bin/activate && goatlab ingest-core

build:
	. .venv/bin/activate && goatlab build-features

train:
	. .venv/bin/activate && goatlab train-models

test:
	. .venv/bin/activate && pytest -q

app:
	. .venv/bin/activate && streamlit run app/Home.py

all: ingest build train test

clean:
	rm -rf data/interim/* data/processed/* models/*
	touch data/interim/.gitkeep data/processed/.gitkeep models/.gitkeep

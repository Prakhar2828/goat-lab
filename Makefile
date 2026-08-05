.PHONY: setup ingest build train test lint app all clean verify-release release-assets release-qa docker-build docker-run

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

lint:
	. .venv/bin/activate && ruff check src tests app scripts

verify-release:
	. .venv/bin/activate && python scripts/verify_public_release.py

release-assets:
	. .venv/bin/activate && python scripts/build_release_assets.py

release-qa: lint test verify-release
	. .venv/bin/activate && python scripts/build_release_assets.py --verify
	find app -type f -name '*.py' -print0 | xargs -0 .venv/bin/python -m py_compile

app:
	. .venv/bin/activate && streamlit run app/Home.py

docker-build:
	docker build -t goat-lab:v1 .

docker-run:
	docker run --rm -p 8501:8501 goat-lab:v1

all: ingest build train test

clean:
	rm -rf data/interim/* data/processed/* models/*
	touch data/interim/.gitkeep data/processed/.gitkeep models/.gitkeep

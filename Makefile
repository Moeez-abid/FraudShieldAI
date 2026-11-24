PY ?= .venv/bin/python
# -I isolates the interpreter from the current directory when reading downloaded data.
RUN = $(PY) -I -m

.PHONY: setup data data-messages data-urls train train-baseline train-url eval test clean-processed

setup:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt -e .

data: data-messages data-urls

data-messages:
	$(RUN) fraudshield.data.download --only uci_sms mendeley_smishing
	$(RUN) fraudshield.data.clean_messages
	$(RUN) fraudshield.data.split messages

data-urls:
	$(RUN) fraudshield.data.download --only phiusiil
	$(RUN) fraudshield.data.clean_urls
	$(RUN) fraudshield.data.split urls

train: train-baseline train-url

train-baseline:
	$(RUN) fraudshield.models.baseline

train-url:
	$(RUN) fraudshield.models.url_rf

# Final evaluation on held-out test sets: run once models are frozen.
eval:
	$(RUN) fraudshield.models.baseline --test
	$(RUN) fraudshield.models.url_rf --test

test:
	$(PY) -m pytest -q

clean-processed:
	rm -rf data/interim/* data/processed/*.parquet data/processed/*.npy

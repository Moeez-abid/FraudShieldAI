PY ?= .venv/bin/python
# -I isolates the interpreter from the current directory when reading downloaded data.
RUN = $(PY) -I -m

.PHONY: setup setup-dl data data-messages data-urls train train-baseline train-url train-url-compare \
        train-distilbert thresholds eval test clean-processed

setup:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt -e .

setup-dl:
	uv pip install --python $(PY) -r requirements-dl.txt

data: data-messages data-urls

data-messages:
	$(RUN) fraudshield.data.download --only uci_sms mendeley_smishing
	$(RUN) fraudshield.data.clean_messages
	$(RUN) fraudshield.data.split messages

data-urls:
	$(RUN) fraudshield.data.download --only phiusiil hannousse
	$(RUN) fraudshield.data.clean_urls
	$(RUN) fraudshield.data.split urls

train: train-baseline train-url train-distilbert thresholds

train-baseline:
	$(RUN) fraudshield.models.baseline

# Serving model: PhiUSIIL host-only + Hannousse full URLs (see clean_urls.py)
train-url:
	$(RUN) fraudshield.models.url_rf --view mixed

# Comparison models for the report (full URL = shortcut, host only)
train-url-compare:
	$(RUN) fraudshield.models.url_rf --view full
	$(RUN) fraudshield.models.url_rf --view host

# Colab/CUDA: default batch 16. 8 GB Macs (MPS runs out of memory): DISTILBERT_ARGS="--cpu --batch-size 8 --grad-accum 2"
train-distilbert:
	PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false $(RUN) fraudshield.models.distilbert $(DISTILBERT_ARGS)

# Risk-band thresholds from validation predictions -> models/risk_thresholds.json
thresholds:
	$(RUN) fraudshield.inference.risk

# Final evaluation on held-out test sets: run once models are frozen.
eval:
	$(RUN) fraudshield.models.baseline --test
	$(RUN) fraudshield.models.url_rf --view mixed --test
	$(RUN) fraudshield.models.distilbert --test $(DISTILBERT_ARGS)

test:
	$(PY) -m pytest -q

clean-processed:
	rm -rf data/interim/* data/processed/*.parquet data/processed/*.npy

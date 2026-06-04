#!/usr/bin/env bash

set -e

# This script is executed before running the wrapper.

# Install dependencies required by the TxGemma embedder.

pip install "transformers>=4.51.0"

pip install "accelerate"

pip install "sentencepiece"

pip install "torch"

pip install "huggingface_hub"

pip install "tqdm"
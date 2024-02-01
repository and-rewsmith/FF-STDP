#!/bin/bash
set -e

# Upgrade pip
python -m pip install --upgrade pip

pip install pytest

python -m pytest -rP datasets/tests
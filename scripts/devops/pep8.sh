#!/bin/bash
set -e

# Upgrade pip
python -m pip install --upgrade pip

# Install pycodestyle
pip install pycodestyle

# Run PEP 8 compliance check
pycodestyle ./datasets --max-line-length=120
pycodestyle ./model --max-line-length=120

echo "PEP 8 check passed successfully!"
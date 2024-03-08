#!/bin/bash
set -e

# Upgrade pip
python -m pip install --upgrade pip

python -m model.tests.smoke.smoke_zenke_2a
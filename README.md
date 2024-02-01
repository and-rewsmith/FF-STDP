# Latent Predictive Learning: SNN Implementation

![CI](https://github.com/and-rewsmith/LPL-SNN/actions/workflows/ci.yaml/badge.svg?branch=main)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This implementation of Latent Predictive Learning (LPL) is based on [this paper](https://www.nature.com/articles/s41593-023-01460-y) by Halvagal and Zenke.

## Directory structure

The structure is as follows:
- `/datasets`: contains various data generation and pytorch `Dataset` implementations. The simplest is fig 2A from the above [paper](https://www.nature.com/articles/s41593-023-01460-y)
- `/model`: contains the model code for a SNN with LPL learning rule
- `/benchmarks`: entrypoints for applying various datasets to the model

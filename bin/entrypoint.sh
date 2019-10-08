#!/usr/bin/env bash

cd src

jupytext --to notebook main.md
jupytext --sync main.ipynb

jupyter lab --ip=0.0.0.0 --port=8765 --no-browser --allow-root

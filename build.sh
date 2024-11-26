#!/bin/bash
set -e

cp pyproject.toml poetry.lock omninexus
poetry build -v

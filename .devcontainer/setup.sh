#!/bin/bash

echo ✅ APT GET UPDATE
echo ✅ --------------
apt-get update -y

echo ✅ Install Jupyter notebook and project requirements
echo ✅ --------------------------------------------------
# Use the base image's Python (3.13) rather than pulling in apt's system
# python3 (3.11), so there's only one Python and VS Code can't pick the
# wrong interpreter for the notebook kernel.
python3 -m pip install --user jupyter
python3 -m pip install --user -r requirements.txt

echo ✅ Install cURL
echo ✅ ----------------
apt-get install curl -y
apt-get clean packages

echo ✅ Install Node.js
echo ✅ ----------------
# Install Node.js 24 as MongoDB MCP server needs at least v20 (https://github.com/mongodb-js/mongodb-mcp-server?tab=readme-ov-file#prerequisites)
apt-get install -y --no-install-recommends npm
npm install -g n
n 24
hash -r 
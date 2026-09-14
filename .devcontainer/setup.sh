#!/bin/bash

echo ✅ APT GET UPDATE
echo ✅ --------------
apt-get update -y

echo ✅ Install Jupyter notebook
echo ✅ ------------------------
apt-get install python3 python3-pip -y --no-install-recommends  
apt-get install jupyter-notebook -y --no-install-recommends  

echo ✅ Install Jupyter Python kernel
echo ✅ -----------------------------
# Python Kernel
apt-get install python3-pymongo python3-ipykernel -y --no-install-recommends
apt-get clean packages

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
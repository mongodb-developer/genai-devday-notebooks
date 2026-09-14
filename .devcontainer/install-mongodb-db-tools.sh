#!/bin/bash

# Install MongoDB command line tools

# We need to download MongoDB Database Tools to import data. In Linux we have two architectures:
# Linux identifies as amd64, in MongoDB the file has ARM64 for Mac M1 
# https://fastdl.mongodb.org/tools/db/mongodb-database-tools-ubuntu2404-arm64-100.11.0.tgz
# Linux identifies as aarch64, in MongoDB the file has x86_64 for intel
# https://fastdl.mongodb.org/tools/db/mongodb-database-tools-ubuntu2404-x86_64-100.11.0.tgz

export MONGO_TOOLS_VERSION=100.11.0 
export ARCH=$(dpkg --print-architecture) 
if [ $ARCH = "amd64" ]; then export TARGET_ARCH="x86_64"; else export TARGET_ARCH="arm64"; fi 
export TARGET="mongodb-database-tools-ubuntu2404-${TARGET_ARCH}-${MONGO_TOOLS_VERSION}.deb" 
echo "Installing tools for architecture $ARCH, linux/${TARGET_ARCH}, target file=${TARGET}"
echo "curl https://fastdl.mongodb.org/tools/db/${TARGET}" 
curl "https://fastdl.mongodb.org/tools/db/${TARGET}" --output "${TARGET}" 
apt-get install -y "./${TARGET}" 
rm "./${TARGET}"

# Install mongosh

# https://downloads.mongodb.com/compass/mongosh-2.4.0-linux-x64.tgz
# https://downloads.mongodb.com/compass/mongosh-2.4.0-linux-arm64.tgz

echo "Installing mongosh"
if [ ARCH = "amd64" ]; then export MONGO_SHELL="mongosh-2.4.0-linux-x64"; else export MONGO_SHELL="mongosh-2.4.0-linux-arm64"; fi 
curl "https://downloads.mongodb.com/compass/${MONGO_SHELL}.tgz" --output "${MONGO_SHELL}.tgz" 
tar xvfz "${MONGO_SHELL}.tgz"
rm "./${MONGO_SHELL}.tgz"
cp mongosh-2.4.0-linux-arm64/bin/* /usr/local/bin/
rm -rf "./${MONGO_SHELL}"
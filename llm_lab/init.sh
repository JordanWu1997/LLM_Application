#!/usr/bin/env bash

###########################################################
# Author      : Kuan-Hsien Wu
# Contact     : jordankhwu@gmail.com
# Datetime    : 2025-02-02 19:56:30
# Description :
###########################################################

# Init for Ollama and Open-WebUI
mkdir -p ./ollama/data
mkdir -p ./ollama/models
mkdir -p ./ollama/open-webui
chmod -R 777 ./ollama

# Init for AnythingLLM
mkdir -p ./anythingllm/data
touch ./anythingllm/.env
chmod -R 777 ./anythingllm/

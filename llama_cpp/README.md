# README

## Run llamacpp-server inside docker

- Docker image: `ghcr.io/ggml-org/llama.cpp:server-cuda`
- Note:
  - All the binaries and libs are stored under `app`
  - If you want to run llama-server manually, please make sure you `cd /app` first

## llamacpp w/ OpenAI API

- Get running model
  ```sh
  curl http://localhost:<PORT>/models
  ```
- Get running model properties
  ```sh
  curl http://localhost:<PORT>/props
  ```

## Connect to running Open-WebUI

- Admin Panel -> Connections -> OpenAI API
  ```
  http://host.docker.internal:<PORT>/v1
  ```

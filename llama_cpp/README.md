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

## `llama-server` Arguments Tuning

- References
  - https://www.koc.com.tw/archives/642193
- Arguments
  - `--n-cpu-moe <layer_number></layer_number>`: lower layer_number -> more GPU usage
  - `--no-mmap`: load model into memory instead of virtual mapping
  - `--chat-template-kwargs '{"enable_thinking":true}'`: enable thinking
  - `--chat-template-kwargs '{"enable_thinking":false}'`: disable thinking

## Connect to running Open-WebUI

- Admin Panel -> Connections -> OpenAI API
  ```
  http://host.docker.internal:<PORT>/v1
  ```

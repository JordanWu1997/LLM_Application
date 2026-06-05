# README

## Setup

- First time usage, add `.env` file in this directory
  ```
  HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN" # Change to your huggingface token
  TOKENIZERS_PARALLEISM="false"
  ```

## Usage

### Download Huggingface Model to Local and Load it to vLLM Server

```sh
model="<MODEL_DOWNLOADED_DIRECTORY>"

export HF_HUB_OFFLINE=1

python3 -m vllm.entrypoints.openai.api_server
  --model "$model" \
  ...
```

### vLLM python

- Ask VLM to describe what you see in Chinese.
  ```sh
  python narrator_stream_vllm.py 0 --mode scene --instruction "用中文回答你看到了什麼?"
  ```

## vLLM Server + vLLM Client

1. Start vLLM server
  1. Run `manage_internal_vllm.sh start` to start server
  2. Run `manage_internal_vllm.sh logs` to check logs
  2. After service is up, run `manage_internal_vllm.sh test` to check if server is running
2. Run
  ```sh
  python narrator_stream_vllm_client.py 0 --mode scene --instruction "用中文回答你看到了什麼?"
  ```

## TODO

- Motion Trigger to save computation time
- Anomaly Detection
- Person Interaction
- Long-Term State Memory for Tracking Object

# README

- First time usage, add `.env` file in this directory
  ```
  HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN" # Change to your huggingface token
  TOKENIZERS_PARALLEISM="false"
  ```

- Example Usage: Ask VLM to describe what you see in Chinese.
  ```sh
  python narrator_stream_vllm.py 0 --mode scene --instruction "用中文回答你看到了什麼?"
  ```

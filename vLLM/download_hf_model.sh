#!/usr/bin/env bash
# vim: set fileencoding=utf-8

# 1. Enforce required positional argument
if [ -z "$1" ]; then
    echo "❌ Error: Missing required argument."
    echo "Usage: $0 <model_id>"
    echo "Example: $0 meta-llama/Meta-Llama-3-8B"
    exit 1
fi

MODEL_ID="$1"
# Extract just the model name for the folder (e.g., "Meta-Llama-3-8B")
MODEL_NAME=${MODEL_ID##*/}
# Target directory on your host machine
DOWNLOAD_DIR="$(pwd)/models/$MODEL_NAME"

echo "======================================================="
echo " 📥 Standalone Model Downloader (using 'hf')"
echo "======================================================="

# 2. Check if the new 'hf' CLI is installed
if ! command -v hf &> /dev/null; then
    echo "⚙️  'hf' CLI not found. Installing Hugging Face CLI..."
    pip install -U "huggingface_hub[cli]"
fi

# 3. Load HF_TOKEN from .env file if available and not already set
if [ -z "$HF_TOKEN" ] && [ -f .env ]; then
    echo "📝 Found .env file, checking for HF_TOKEN..."
    # Extracts value, strips spaces, and removes single or double quotes
    HF_TOKEN=$(grep -E '^HF_TOKEN=' .env | sed -E 's/^HF_TOKEN=[[:space:]]*["'\'']?//;s/["'\'']?[[:space:]]*$//')
    if [ -n "$HF_TOKEN" ]; then
        echo "✅ Loaded HF_TOKEN from .env"
        export HF_TOKEN
    fi
fi

# 4. Smart check: Does this model require authentication?
echo "🔍 Checking repository access for '$MODEL_ID'..."

if ! python3 -c "from huggingface_hub import model_info; model_info('$MODEL_ID', token='${HF_TOKEN:-}')" &> /dev/null; then
    echo "⚠️  This model is gated/private (or requires authentication)."

    # If it wasn't in the environment or .env, ask the user interactively
    if [ -z "$HF_TOKEN" ]; then
        read -s -p "🔑 Enter your HF Token (starts with hf_): " HF_TOKEN
        echo ""
        export HF_TOKEN
    fi

    if ! python3 -c "from huggingface_hub import model_info; model_info('$MODEL_ID', token='$HF_TOKEN')" &> /dev/null; then
        echo "❌ Error: Access denied. Either the token is invalid, or you haven't accepted the model's license agreement on Hugging Face."
        exit 1
    fi
    echo "✅ Token accepted!"
else
    echo "✅ Public model detected. No token required."
fi

echo "🚀 Starting download for: $MODEL_ID"
echo "📁 Destination: $DOWNLOAD_DIR"

# 5. Execute Download using the new 'hf' CLI
hf download "$MODEL_ID" \
    --local-dir "$DOWNLOAD_DIR" \
    --exclude "*.md" \
    --exclude ".gitattributes"

echo "======================================================="
echo " ✅ Download Complete!"
echo " 📦 Your model is fully self-contained in: $DOWNLOAD_DIR"
echo "======================================================="

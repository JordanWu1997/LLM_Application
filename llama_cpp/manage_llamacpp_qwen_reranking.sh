#!/usr/bin/env bash

# Configurable defaults - pre-tuned for Qwen-0.6B-Embed inside a 1.5GB VRAM ceiling
DEFAULT_MODEL="/workplace/models/Qwen3-Reranker-0.6B-GGUF-Voodisss-Q8/Qwen3-Reranker-0.6B.Q8_0.gguf"
DEFAULT_CTX=2048          # 4096 is the sweet spot for 1.5GB VRAM (8192 risks OOM on long batches)
DEFAULT_BATCH=1024        # Max tokens processed in logical sequence
DEFAULT_UBATCH=256        # Physical VRAM compute chunk-size (Keeps VRAM spikes flat)
DEFAULT_NGL=999           # 100% GPU offload (Model is ~600MB, fits easily)
PORT=8083

PID_FILE="/tmp/llamacpp_embed_server.pid"
LOG_FILE="/tmp/llamacpp_embed_server.log"

print_usage() {
    echo "======================================================="
    echo " 📐 Internal llama.cpp Reranking Controller"
    echo "======================================================="
    echo "Usage: $0 [start | stop | restart | status | logs | test]"
    echo "======================================================="
}

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi

    # Port-specific check to avoid grabbing your Gemma chat server's PID
    local stray_pid=$(pgrep -f "llama-server.*--port $PORT")
    if [ -n "$stray_pid" ]; then
        echo "$stray_pid" > "$PID_FILE"
        return 0
    fi

    return 1
}

stop_server() {
    echo "🔄 Stopping internal Reranking server..."

    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        echo "   -> Sending graceful stop to strict PID: $pid"
        kill -15 "$pid" 2>/dev/null

        for i in {1..10}; do
            if ! ps -p "$pid" > /dev/null 2>&1; then
                echo "🛑 Server stopped cleanly. VRAM freed."
                rm -f "$PID_FILE"
                return 0
            fi
            sleep 1
        done

        echo "⚠️  Server taking too long. Forcing termination..."
        kill -9 "$pid" 2>/dev/null
        rm -f "$PID_FILE"
        echo "💀 Server completely terminated."
    else
        echo "ℹ️  No PID file found. Checking for orphaned process on port $PORT..."
        local stray=$(pgrep -f "llama-server.*--port $PORT")
        if [ -n "$stray" ]; then
            kill -9 "$stray" 2>/dev/null
            echo "🛑 Orphaned embedding server killed."
        else
            echo "ℹ️  Nothing to clean up."
        fi
    fi
}

start_server() {
    if is_running; then
        echo "⚠️  Reranking server is already running. Stop it first."
        exit 0
    fi

    local model="${MODEL:-$DEFAULT_MODEL}"
    local ctx="${CTX_SIZE:-$DEFAULT_CTX}"
    local ngl="${N_GPU_LAYERS:-$DEFAULT_NGL}"

    echo "🚀 Launching internal llama-server (Reranking Mode)..."

    cd /app || exit 1

    ./llama-server \
        --model "$model" \
        --reranking \
        --n-gpu-layers "$ngl" \
        --ctx-size "$ctx" \
        --batch-size "$DEFAULT_BATCH" \
        --ubatch-size "$DEFAULT_UBATCH" \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        --cache-ram 0 \
        --flash-attn on \
        --host 0.0.0.0 \
        --port "$PORT" \
        > "$LOG_FILE" 2>&1 &

    local strict_pid=$!
    echo "$strict_pid" > "$PID_FILE"
    echo "✅ Process locked and tracked with strict PID: $strict_pid"
}

test_server() {
    if ! is_running; then
        echo "❌ Error: The embedding server is not running."
        exit 1
    fi

    echo -n "📡 [Step 1/2] Pinging Endpoint via curl... "
    local curl_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/v1/models)

    if [ "$curl_status" -eq 200 ]; then
        echo "🟢 SUCCESS (HTTP 200)"
    else
        echo "🔴 FAILED (HTTP Status: $curl_status)"
        exit 1
    fi

    echo "🧠 [Step 2/2] Submitting vector tensor test payload..."

    cat << 'EOF' > /tmp/embed_test.py
import urllib.request
import json
import sys

url = f"http://localhost:{sys.argv[1]}/v1/embeddings"
headers = {"Content-Type": "application/json"}
payload = {
    "model": "qwen3-embedding",
    "input": "The quick brown fox jumps over the lazy dog."
}

try:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode("utf-8"))
        vector = res['data'][0]['embedding']
        dim = len(vector)
        print(f"   🎉 Vector Generation Clear!")
        print(f"   📐 Tensor Dimensions : {dim}")
        print(f"   🔢 Sample Coordinates: [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}, ...]")
        sys.exit(0)
except Exception as e:
    print(f"   ❌ Python Reranking Test Failed -> {e}")
    sys.exit(1)
EOF

    python3 /tmp/embed_test.py "$PORT"
    local test_result=$?
    rm /tmp/embed_test.py

    if [ $test_result -eq 0 ]; then
        echo "🟢 ALL TESTS PASSED! Reranking Engine is healthy."
    else
        echo "❌ DIAGNOSTICS FAILED!"
        exit 1
    fi
}

case "$1" in
    start) start_server ;;
    stop) stop_server ;;
    restart) stop_server; sleep 1; start_server ;;
    status)
        if is_running; then
            echo "🟢 llama.cpp Embed Server Status: RUNNING (Port $PORT)"
        else
            echo "🔴 llama.cpp Embed Server Status: DOWN"
        fi
        ;;
    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "❌ No log file found at $LOG_FILE yet."
        fi
        ;;
    test) test_server ;;
    *)
        print_usage
        exit 1
        ;;
esac

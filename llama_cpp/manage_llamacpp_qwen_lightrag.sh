#!/usr/bin/env bash

DEFAULT_MODEL="/workplace/models/Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-UD-Q5_K_XL.gguf"
DEFAULT_MMPROJ_MODEL="/workplace/models/Qwen3.5-9B-MTP-GGUF/mmproj-BF16.gguf"

#DEFAULT_CTX=131072
#DEFAULT_CTX=65536
#DEFAULT_CTX=32768
DEFAULT_CTX=16384
DEFAULT_NGL=999
PORT=8081

PID_FILE="/tmp/llamacpp_server.pid"
LOG_FILE="/tmp/llamacpp_server.log"

print_usage() {
    echo "======================================================="
    echo " 🦙 Internal llama.cpp Process Controller"
    echo "======================================================="
    echo "Usage: $0 [start | stop | restart | status | logs | test]"
    echo "======================================================="
}

is_running() {
    # 1. Primary Check: Strict PID validation
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi

    # 2. Safe Fallback: pidof avoids matching this bash script's name
    if pidof llama-server > /dev/null 2>&1; then
        pidof llama-server | awk '{print $1}' > "$PID_FILE"
        return 0
    fi

    return 1
}

stop_server() {
    echo "🔄 Stopping internal llama.cpp server..."

    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        echo "   -> Sending graceful stop to strict PID: $pid"
        kill -15 "$pid" 2>/dev/null

        # Wait up to 15 seconds for mmap RAM to cleanly unmap
        for i in {1..15}; do
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
        echo "ℹ️  No PID file found. Cleaning up stray processes..."
        for stray in $(pidof llama-server); do
            kill -9 "$stray" 2>/dev/null
        done
        echo "🛑 Cleaned up."
    fi
}

start_server() {
    if is_running; then
        echo "⚠️  llama.cpp server is already running. Stop it first."
        exit 0
    fi

    local model="${MODEL:-$DEFAULT_MODEL}"
    local mmproj_model="${MODEL:-$DEFAULT_MMPROJ_MODEL}"
    local draft_model="${MODEL:-$DEFAULT_DRAFT_MODEL}"
    local ctx="${CTX_SIZE:-$DEFAULT_CTX}"
    local ngl="${N_GPU_LAYERS:-$DEFAULT_NGL}"

    echo "🚀 Launching internal llama-server..."

    # Must run from /app to find shared libraries
    cd /app || exit 1

    # Run in background (GPU: 3060 (vRAM 12GB) + MOE CPU offload)
    # NOTE:
    #   --n-cpu-moe layer_number: lower layer_number -> more GPU usage
    #   --no-mmap: load model into memory instead of virtual mapping
    #   --chat-template-kwargs '{"enable_thinking":true}': enable thinking
    #   --chat-template-kwargs '{"enable_thinking":false}': disable thinking
    #   --jinja: avoid template issue for new qwen model

    ./llama-server \
        --model "$model" \
        --ctx-size "$ctx" \
        --spec-type draft-mtp \
        --spec-draft-n-max 4 \
        --parallel 1 \
        -b 2048 \
        -ub 512 \
        --n-gpu-layers "$ngl" \
        --flash-attn on \
        -t 6 \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        --cache-type-k-draft q8_0 \
        --cache-type-v-draft q8_0 \
        --cache-ram 2048 \
        --temp 0.95 \
        --top-p 0.95 \
        --top-k 20 \
        --min-p 0.0 \
        --presence-penalty 1.5 \
        --repeat-penalty 1.0 \
        -n 8192 \
        --reasoning-budget 4096 \
        --host 0.0.0.0 \
        --port "$PORT" \
        --jinja \
        --chat-template-kwargs '{"enable_thinking": true}' \
        > "$LOG_FILE" 2>&1 &

    # The magic bullet: Grab the exact PID of the last background command
    local strict_pid=$!
    echo "$strict_pid" > "$PID_FILE"
    echo "✅ Process locked and tracked with strict PID: $strict_pid"
}

test_server() {
    if ! is_running; then
        echo "❌ Error: The server is not running."
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

    echo "🧠 [Step 2/2] Submitting reasoning test payload..."

    cat << 'EOF' > /tmp/llama_test.py
import urllib.request
import json
import sys

url = "http://localhost:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
    "max_tokens": 10
}

try:
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
        content = result['choices'][0]['message']['content'].strip()
        print(f"   🎉 Inference Test Clear!")
        print(f"   🤖 Model Response: \"{content}\"")
        sys.exit(0)
except Exception as e:
    print(f"   ❌ Python Inference Test Failed -> {e}")
    sys.exit(1)
EOF

    python3 /tmp/llama_test.py
    local test_result=$?
    rm /tmp/llama_test.py

    if [ $test_result -eq 0 ]; then
        echo "🟢 ALL TESTS PASSED! Server is healthy."
    else
        echo "❌ DIAGNOSTICS FAILED!"
        exit 1
    fi
}

case "$1" in
    start) start_server ;;
    stop) stop_server ;;
    restart) stop_server; sleep 2; start_server ;;
    status)
        if is_running; then
            echo "🟢 llama.cpp Server Status: RUNNING"
        else
            echo "🔴 llama.cpp Server Status: DOWN / STOPPED"
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

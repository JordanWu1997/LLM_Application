#!/usr/bin/env bash

# Configurable defaults - pre-tuned for Gemma-4-26B with MoE RAM offloading
DEFAULT_MODEL="/workplace/models/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
DEFAULT_CTX=32768
DEFAULT_NGL=99
DEFAULT_BATCH=1024
PORT=8080

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
    # 1. Primary check: Verify tracked file PID
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi

    # 2. Resilient fallback check: Look for any running llama-server process
    if pgrep -f "llama-server" > /dev/null 2>&1; then
        # Dynamically repair and log the real process ID back into track file
        pgrep -f "llama-server" | head -n 1 > "$PID_FILE"
        return 0
    fi

    return 1
}

stop_server() {
    if is_running; then
        local pids=$(pgrep -f "llama-server")
        if [ -z "$pids" ] && [ -f "$PID_FILE" ]; then
            pids=$(cat "$PID_FILE")
        fi

        echo "🔄 Stopping internal llama.cpp server processes..."
        for pid in $pids; do
            kill -15 "$pid" 2>/dev/null
        done

        for i in {1..15}; do
            if ! pgrep -f "llama-server" > /dev/null 2>&1; then
                rm -f "$PID_FILE"
                echo "🛑 llama.cpp server stopped cleanly. VRAM freed."
                return 0
            fi
            sleep 1
        done

        echo "⚠️  llama.cpp server didn't stop in time. Forcing termination..."
        pgrep -f "llama-server" | xargs kill -9 2>/dev/null
        rm -f "$PID_FILE"
        echo "💀 llama.cpp server force killed."
    else
        echo "ℹ️  No active internal llama.cpp server process detected."
        rm -f "$PID_FILE"
    fi
}

start_server() {
    if is_running; then
        echo "⚠️  llama.cpp server is already running. Stop it first or run 'restart'."
        exit 0
    fi

    local model="${MODEL:-$DEFAULT_MODEL}"
    local ctx="${CTX_SIZE:-$DEFAULT_CTX}"
    local ngl="${N_GPU_LAYERS:-$DEFAULT_NGL}"
    local batch="${BATCH_SIZE:-$DEFAULT_BATCH}"

    echo "🚀 Launching internal llama-server in background..."
    echo "📌 Model: $model"
    echo "📌 Context: $ctx | GPU Layers: $ngl | MoE Offload: CPU"

    > "$LOG_FILE"

    # Navigate to the app directory so it can find its shared libraries
    cd /app || exit 1

    # Run server background job with optimized MoE/Context flags
    ./llama-server \
        -m "$model" \
        -c "$ctx" \
        -ngl "$ngl" \
        -b "$batch" \
        -ot "exps=CPU" \
        -fa on \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        --host 0.0.0.0 \
        --port "$PORT" \
        > "$LOG_FILE" 2>&1 &

    # Give process breathing room to allocate initial tensor memory
    sleep 2.0

    local real_pid=$(pgrep -f "llama-server" | head -n 1)
    if [ -n "$real_pid" ]; then
        echo "$real_pid" > "$PID_FILE"
        echo "✅ Process tracked successfully with PID: $real_pid"
    else
        echo "⚠️  Process spawned, tracking file setup deferred to auto-discovery."
    fi
}

test_server() {
    echo "======================================================="
    echo " 🧪 Running Diagnostics on Internal llama.cpp Engine"
    echo "======================================================="

    if ! is_running; then
        echo "❌ Error: The llama-server background process is not running."
        exit 1
    fi

    echo -n "📡 [Step 1/2] Pinging Endpoint via curl... "
    local curl_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/v1/models)

    if [ "$curl_status" -eq 200 ]; then
        echo "🟢 SUCCESS (HTTP 200)"
    else
        echo "🔴 FAILED (HTTP Status: $curl_status)"
        echo "   The server might still be allocating KV cache. Check logs."
        exit 1
    fi

    echo "🧠 [Step 2/2] Submitting reasoning test payload..."

    # Create a temporary python script to test the endpoint
    cat << 'EOF' > /tmp/llama_test.py
import urllib.request
import json
import sys

url = "http://localhost:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "gemma-4-26b",
    "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
    "max_tokens": 10
}

try:
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))

        # Grab final content (ignoring reasoning channel for this quick test)
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
        echo "======================================================="
        echo " 🟢 ALL TESTS PASSED! Server is healthy and responsive."
        echo "======================================================="
    else
        echo "======================================================="
        echo " ❌ DIAGNOSTICS FAILED! Look over your internal logs."
        echo "======================================================="
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

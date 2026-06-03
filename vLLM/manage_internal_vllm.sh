#!/usr/bin/env bash

# Configurable defaults - edit these or pass them dynamically
DEFAULT_MODEL="cyankiwi/Qwen3.5-4B-AWQ-4bit"
DEFAULT_QUANTIZATION="compressed-tensors"  # Updated to match model config
DEFAULT_MAX_MODEL_LEN=2048
DEFAULT_GPU_UTIL=0.75
DEFAULT_MAX_SEQS=2
PORT=8000

PID_FILE="/tmp/vllm_server.pid"
LOG_FILE="/tmp/vllm_server.log"

print_usage() {
    echo "======================================================="
    echo " ⚙️  Internal vLLM Process Controller"
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

    # 2. Resilient fallback check: Look for any running vllm api_server process
    if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null 2>&1; then
        # Dynamically repair and log the real process ID back into track file
        pgrep -f "vllm.entrypoints.openai.api_server" | head -n 1 > "$PID_FILE"
        return 0
    fi

    return 1
}

stop_server() {
    if is_running; then
        # Dynamically pull matching engine targets
        local pids=$(pgrep -f "vllm.entrypoints.openai.api_server")
        if [ -z "$pids" ] && [ -f "$PID_FILE" ]; then
            pids=$(cat "$PID_FILE")
        fi

        echo "🔄 Stopping internal vLLM server processes..."
        for pid in $pids; do
            kill -15 "$pid" 2>/dev/null
        done

        for i in {1..15}; do
            if ! pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null 2>&1; then
                rm -f "$PID_FILE"
                echo "🛑 vLLM server stopped cleanly."
                return 0
            fi
            sleep 1
        done

        echo "⚠️  vLLM server didn't stop in time. Forcing termination..."
        pgrep -f "vllm.entrypoints.openai.api_server" | xargs kill -9 2>/dev/null
        rm -f "$PID_FILE"
        echo "💀 vLLM server force killed."
    else
        echo "ℹ️  No active internal vLLM server process detected."
        rm -f "$PID_FILE"
    fi
}

start_server() {
    if is_running; then
        echo "⚠️  vLLM server is already running. Stop it first or run 'restart'."
        exit 0
    fi

    local model="${MODEL:-$DEFAULT_MODEL}"
    local quant="${QUANTIZATION:-$DEFAULT_QUANTIZATION}"
    local max_len="${MAX_MODEL_LEN:-$DEFAULT_MAX_MODEL_LEN}"
    local gpu_util="${GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_UTIL}"
    local max_seqs="${MAX_NUM_SEQS:-$DEFAULT_MAX_SEQS}"

    echo "🚀 Launching internal vLLM python module in background..."
    echo "📌 Model: $model"
    echo "📌 Quantization: $quant"

    > "$LOG_FILE"

    export PYTHONUNBUFFERED=1

    # Run server background job
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --quantization "$quant" \
        --max-model-len "$max_len" \
        --gpu-memory-utilization "$gpu_util" \
        --max-num-seqs "$max_seqs" \
        --enforce-eager \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        2>&1 | tee "$LOG_FILE" > /proc/1/fd/1 &

    # Give process structural breathing room to spawn parent PID cleanly
    sleep 1.5

    # Capture the true python process ID across pipelined tee
    local real_pid=$(pgrep -f "vllm.entrypoints.openai.api_server" | head -n 1)
    if [ -n "$real_pid" ]; then
        echo "$real_pid" > "$PID_FILE"
        echo "✅ Process tracked successfully with PID: $real_pid"
    else
        echo "⚠️  Process spawned, tracking file setup deferred to auto-discovery."
    fi
}

test_server() {
    echo "======================================================="
    echo " 🧪 Running Diagnostics on Internal vLLM Engine"
    echo "======================================================="

    if ! is_running; then
        echo "❌ Error: The vLLM background process is not running."
        exit 1
    fi

    echo -n "📡 [Step 1/2] Pinging vLLM Endpoint via curl... "
    local curl_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/v1/models)

    if [ "$curl_status" -eq 200 ]; then
        echo "🟢 SUCCESS (HTTP 200)"
    else
        echo "🔴 FAILED (HTTP Status: $curl_status)"
        echo "   The server might still be allocating VRAM checkpoints. Check logs."
        exit 1
    fi

    echo "🧠 [Step 2/2] Submitting vision test payload via OpenAI Client..."

    python3 - <<EOF
import base64
import io
import sys
from openai import OpenAI

try:
    mock_base64_image = "R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
    client = OpenAI(base_url="http://localhost:${PORT}/v1", api_key="test-token")

    models = client.models.list()
    active_model = models.data[0].id
    print(f"   Connected to active engine target: {active_model}")

    response = client.chat.completions.create(
        model=active_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image? Reply in one word. /no_think"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/gif;base64,{mock_base64_image}"}
                    }
                ]
            }
        ],
        max_tokens=10,
        extra_body={"enable_thinking": False}
    )

    output_text = response.choices[0].message.content.strip()
    print(f"   🎉 VLM Processing Test Clear!")
    print(f"   🤖 Model Response: \"{output_text}\"")
    sys.exit(0)
except Exception as e:
    print(f"   ❌ Python Inference Test Failed -> {e}")
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
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
            echo "🟢 vLLM Server Status: RUNNING"
        else
            echo "🔴 vLLM Server Status: DOWN / STOPPED"
        fi
        ;;
    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "❌ No log file found at $LOG_FILE yet."
        fi
        ;;
    test)
        test_server
        ;;
    *)
        print_usage
        exit 1
        ;;
esac

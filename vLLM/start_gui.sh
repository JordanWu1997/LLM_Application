#!/usr/bin/env bash

# 0. Parse command-line arguments
REBUILD=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--rebuild) REBUILD=1; shift ;;
        -h|--help)
            echo "Usage: $0 [-r|--rebuild]"
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Usage: $0 [-r|--rebuild]"
            exit 1
            ;;
    esac
done

echo "Preparing X11 forwarding for VNC Host Display ($DISPLAY)..."

# 1. Create a wildcard Xauthority cookie to bypass hostname mismatches
export XAUTH_FILE=/tmp/.docker.xauth
rm -f "$XAUTH_FILE"
touch "$XAUTH_FILE"
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH_FILE" nmerge -
chmod 777 "$XAUTH_FILE"

# 2. Run Docker Compose
# (Compose will automatically read the exported DISPLAY and XAUTH_FILE variables)
echo "Starting pipeline via Docker Compose..."

if [ "$REBUILD" -eq 1 ]; then
    echo "Rebuild flag detected. Force-rebuilding image..."
    docker compose build --no-cache
    docker compose up -d --force-recreate
else
    echo "Starting standard container..."
    docker compose up -d
fi

echo "--------------------------------------------------------"
echo "Container is running in the background!"
echo "You can safely close this terminal shell now."
echo "To view logs, run: docker compose logs -f"
echo "To stop the container, run: ./stop_gui.sh"
echo "--------------------------------------------------------"

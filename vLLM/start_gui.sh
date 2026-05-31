#!/usr/bin/env bash

echo "Preparing X11 forwarding for VNC Host Display ($DISPLAY)..."

# 1. Create a wildcard Xauthority cookie to bypass hostname mismatches
export XAUTH_FILE=/tmp/.docker.xauth
rm -f $XAUTH_FILE
touch $XAUTH_FILE
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH_FILE nmerge -
chmod 777 $XAUTH_FILE

# 2. Run Docker Compose
# (Compose will automatically read the exported DISPLAY and XAUTH_FILE variables)
echo "Starting pipeline via Docker Compose..."
docker compose up -d --build

echo "--------------------------------------------------------"
echo "Container is running in the background!"
echo "You can safely close this terminal shell now."
echo "To view logs, run: docker compose logs -f"
echo "To stop the container, run: ./stop_gui.sh"
echo "--------------------------------------------------------"

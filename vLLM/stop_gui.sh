#!/usr/bin/env bash

echo "Stopping container environment..."

# If XAUTH_FILE is empty, temporarily set it to a dummy path just
# to pass the docker-compose syntax check during shutdown.
export XAUTH_FILE=${XAUTH_FILE:-/tmp/.docker.xauth_dummy}
touch $XAUTH_FILE

docker compose down

echo "Cleaning up local X11 cookie..."
rm -f "$(pwd)/.docker.xauth"

echo "Environment stopped successfully."

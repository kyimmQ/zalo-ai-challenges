#!/bin/bash

set -e

if [ ! -f "roadbuddy.tar.gz" ]; then
    echo "Error: roadbuddy.tar.gz not found!"
    echo "Run build_docker.sh first"
    exit 1
fi

echo "Loading Docker image..."
docker load < roadbuddy.tar.gz

echo "Testing Docker container..."
docker run --rm --gpus all \
    -v "$(pwd)/data:/data" \
    -v "$(pwd)/docker_output:/output" \
    roadbuddy:latest \
    --test_json /data/public_test/public_test.json \
    --video_dir /data \
    --output_csv /output/submission.csv \
    --batch_size 8

echo ""
echo "Test completed!"
echo "Check docker_output/submission.csv for results"

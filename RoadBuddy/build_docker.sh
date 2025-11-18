#!/bin/bash

set -e

echo "Building RoadBuddy Docker image..."

if [ ! -f "best_model.pth" ]; then
    echo "Error: best_model.pth not found!"
    echo "Please copy your best checkpoint to best_model.pth"
    echo "Example: cp ckpts/phase2/best.pth best_model.pth"
    exit 1
fi

docker build -t roadbuddy:latest .

echo "Saving Docker image..."
docker save roadbuddy:latest | gzip > roadbuddy.tar.gz

echo "Computing checksum..."
if command -v md5sum &> /dev/null; then
    md5sum roadbuddy.tar.gz > roadbuddy.tar.gz.md5
    cat roadbuddy.tar.gz.md5
elif command -v md5 &> /dev/null; then
    md5 roadbuddy.tar.gz > roadbuddy.tar.gz.md5
    cat roadbuddy.tar.gz.md5
fi

echo ""
echo "Docker image built successfully!"
echo "File: roadbuddy.tar.gz"
echo "Size: $(du -h roadbuddy.tar.gz | cut -f1)"
echo ""
echo "Upload this file to Google Drive and submit the link to Zalo AI Challenge portal"

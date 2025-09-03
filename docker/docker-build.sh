#!/bin/bash

# SwapTube Docker Build Script
# This script builds the SwapTube Docker container
# Run from the docker/ directory

set -e

# Check if we're in the docker directory
if [ ! -f "Dockerfile" ]; then
    echo "❌ Error: Please run this script from the docker/ directory"
    echo "Usage: cd docker && ./docker-build.sh"
    exit 1
fi

echo "🐳 Building SwapTube Docker container..."

# Build the Docker image
docker-compose build swaptube

echo "✅ SwapTube Docker container built successfully!"
echo "📺 Render a project with: ./docker-render.sh ProjectName 640 360"
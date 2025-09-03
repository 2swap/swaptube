#!/bin/bash

# SwapTube Docker Render Script
# This script renders a SwapTube project using Docker
# Run from the docker/ directory

set -e

# Check if we're in the docker directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: Please run this script from the docker/ directory"
    echo "Usage: cd docker && ./docker-render.sh ProjectName Width Height [options]"
    exit 1
fi

# Check if project name is provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <ProjectName> <Width> <Height> [options]"
    echo "Example: $0 Demos/LoopingLambdaDemo 640 360"
    echo "Example: $0 MyProject 1920 1080 -s  # smoketest"
    exit 1
fi

PROJECT_NAME=$1
WIDTH=$2
HEIGHT=$3
shift 3  # Remove first 3 arguments
EXTRA_ARGS="$@"  # Remaining arguments (like -s for smoketest)

echo "🎬 Rendering SwapTube project: $PROJECT_NAME"
echo "📐 Resolution: ${WIDTH}x${HEIGHT}"
if [ ! -z "$EXTRA_ARGS" ]; then
    echo "⚙️  Extra options: $EXTRA_ARGS"
fi

# Ensure the container is built
if ! docker images | grep -q swaptube; then
    echo "🔨 Docker image not found. Building..."
    ./docker-build.sh
fi

# Run the render
echo "🚀 Starting render..."
docker-compose run --rm swaptube ./go.sh "$PROJECT_NAME" "$WIDTH" "$HEIGHT" $EXTRA_ARGS

echo "✅ Render complete! Check the ../out/ directory for results."
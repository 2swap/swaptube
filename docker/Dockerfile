# SwapTube Docker Container
# Based on Ubuntu 22.04 for better compatibility with CUDA and modern libraries
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    # FFMPEG and development libraries
    ffmpeg \
    libswscale-dev \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    # Graphics and visualization
    gnuplot \
    libglm-dev \
    librsvg2-dev \
    libglib2.0-dev \
    libcairo2-dev \
    libpng-dev \
    # Json library
    nlohmann-json3-dev \
    # Python for recording scripts
    python3 \
    python3-pip \
    # CUDA toolkit (optional - will install if available)
    nvidia-cuda-toolkit \
    # Audio libraries
    libasound2-dev \
    pulseaudio \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /workspace

# Clone and build MicroTeX
RUN git clone https://github.com/NanoMichael/MicroTeX.git MicroTeX-master \
    && cd MicroTeX-master \
    && mkdir -p build \
    && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && cd /workspace

# Create necessary directories
RUN mkdir -p /workspace/swaptube/out \
    && mkdir -p /workspace/swaptube/media \
    && mkdir -p /workspace/swaptube/build

# Set environment variables
ENV PATH="/workspace/MicroTeX-master/build:$PATH"
ENV MICROTEX_ROOT="/workspace/MicroTeX-master"

# Copy the swaptube source code
COPY .. /workspace/swaptube/

# Set working directory to swaptube
WORKDIR /workspace/swaptube

# Make scripts executable
RUN chmod +x go.sh

# Create a non-root user for running the application
RUN useradd -m -s /bin/bash swaptube \
    && chown -R swaptube:swaptube /workspace

# Switch to non-root user
USER swaptube

# Expose any ports if needed (for development/debugging)
EXPOSE 8080

# Default command
CMD ["/bin/bash"]

# ------------------------------------------------------------
# SwapTube: CUDA + FFmpeg5 on Debian 12 (bookworm)
# Requires: NVIDIA Container Toolkit on host, run with --gpus all
# ------------------------------------------------------------
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC
ENV CUDA_HOME=/usr/local/cuda \
    CUDAHOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PATH=/usr/local/cuda/bin:${PATH}


# Base build + deps from README (FFmpeg 5.x native on bookworm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl pkg-config ca-certificates \
    ffmpeg \
    libswscale-dev libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev \
    gnuplot \
    libglm-dev \
    librsvg2-dev libglib2.0-dev libcairo2-dev \
    libeigen3-dev \
    libpng-dev \
    python3 python3-pip \
    # Json library
    nlohmann-json3-dev \
    libtinyxml2-dev \
    libgtkmm-3.0-dev \
  && rm -rf /var/lib/apt/lists/*

# Create app directories
WORKDIR /workspace
# Expect you to docker build from the repo root so "." is the swaptube checkout
COPY .. /workspace

# -------- MicroTeX --------
# Build MicroTeX into /opt/MicroTeX-master and expose it as a sibling of /workspace via symlink.
# This satisfies "install MicroTeX in MicroTeX-master alongside the swaptube checkout".
# + Qt6 dev headers for MicroTeX (Qt backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
    qt6-base-dev \
 && rm -rf /var/lib/apt/lists/*

# Build MicroTeX using the Qt backend (avoid GTK/gtksourceviewmm)
RUN git clone --depth=1 https://github.com/NanoMichael/MicroTeX.git /opt/MicroTeX-master \
 && cmake -S /opt/MicroTeX-master -B /opt/MicroTeX-master/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DQT=ON \
 && cmake --build /opt/MicroTeX-master/build -j"$(nproc)" \
 && ln -s /opt/MicroTeX-master /workspace/MicroTeX-master \
 && ln -s /opt/MicroTeX-master /workspace/swaptube/MicroTeX-master


ENV MICROTEX_ROOT=/opt/MicroTeX-master
ENV PATH="/opt/MicroTeX-master/build:${PATH}"

# Make sure scripts are executable
RUN chmod +x /workspace/go.sh || true

# Optional: build the project once so the image has the compiled binary in /workspace/build
# (Leave this commented if you prefer to build at run-time.)
# RUN cmake -S /app -B /app/build -DCMAKE_BUILD_TYPE=Release && cmake --build /workspace/build -j"$(nproc)"

# ---- Create non-root user that matches host uid/gid (override at build-time if needed) ----
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} swaptube && useradd -m -u ${UID} -g ${GID} -s /bin/bash swaptube \
 && chown -R ${UID}:${GID} /workspace /opt/MicroTeX-master

USER swaptube

# Useful mount points
VOLUME ["/workspace/media", "/workspace/out"]

# Default to an interactive shell; you’ll pass ./go.sh ... at docker run time.
WORKDIR /workspace
CMD ["/bin/bash"]


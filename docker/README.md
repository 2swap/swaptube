# SwapTube Docker Setup

This directory contains Docker configuration files and scripts to run SwapTube in a containerized environment, making it easy to deploy and use without installing all dependencies manually.

## Note from 2swap:

I personally don't use this as I have gotten accustomed to building and running SwapTube natively on my system, and I intend to keep it that way to maintain a level of familiarity with external dependencies. This setup is contributed by the community, and I do not intend to maintain it. YMMV.

## 🐳 Quick Start

1. **Build the container:**
   ```bash
   cd docker
   ./docker-build.sh
   ```

2. **Render a project:**
   ```bash
   ./docker-render.sh Demos/LoopingLambdaDemo 640 360
   ```

## 📁 Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Main container definition with all SwapTube dependencies |
| `docker-compose.yml` | Docker Compose configuration for easy container management |
| `docker-build.sh` | Build the SwapTube Docker image |
| `docker-render.sh` | Render a project using Docker |

## 🚀 Usage

### Building the Container

```bash
cd docker
./docker-build.sh
```

This will build a Docker image with all required dependencies:
- FFMPEG and development libraries
- CUDA toolkit (for GPU acceleration)
- GLM, Cairo, RSVG, Eigen3, and other graphics libraries
- MicroTeX (for LaTeX rendering)
- Build tools (CMake, GCC, etc.)

### Rendering Projects

```bash
./docker-render.sh <ProjectName> <Width> <Height> [options]
```

Examples:
```bash
# Basic render
./docker-render.sh Demos/LoopingLambdaDemo 640 360

# Smoketest (fast validation)
./docker-render.sh MyProject 640 360 -s
```

### Development Mode

```bash
docker-compose up -d swaptube
docker-compose exec swaptube /bin/bash
```

### Cleanup

```bash
docker-compose down
```

## 🔧 Configuration

### GPU Support

The container includes NVIDIA GPU support for CUDA acceleration. Make sure you have:
- NVIDIA Docker runtime installed
- Compatible GPU drivers

If you don't have GPU support, the container will still work but without CUDA acceleration.

### Volume Mounts

The Docker setup automatically mounts:
- `../` (SwapTube source code) → `/workspace/swaptube/`
- `../out/` (output directory) → `/workspace/swaptube/out/`
- `../media/` (input media) → `/workspace/swaptube/media/`

Generated videos and files will appear in the host's `out/` directory.

### Environment Variables

You can customize the container behavior by setting environment variables in `docker-compose.yml`:
- `NVIDIA_VISIBLE_DEVICES=all` - Enable all GPUs
- `DISPLAY=${DISPLAY:-:0}` - For GUI applications (if needed)

## 🛠️ Troubleshooting

### Permission Issues
If you encounter permission issues with generated files:
```bash
# Fix ownership of output files
sudo chown -R $USER:$USER ../out/
```

### Container Won't Start
1. Make sure Docker is running
2. Try rebuilding: `./docker-build.sh`
3. Check logs: `docker-compose logs swaptube`

### Missing Dependencies
The Dockerfile should include all required dependencies. If something is missing:
1. Edit `Dockerfile` to add the missing package
2. Rebuild: `./docker-build.sh`

### CUDA Issues
If CUDA isn't working:
1. Verify GPU support: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`
2. Check if `nvidia-docker` is installed
3. The container will fall back to CPU-only mode if GPU isn't available

## 📋 Notes

- All scripts must be run from the `docker/` directory
- Output files are saved to `../out/` (relative to the docker directory)
- The container includes a non-root `swaptube` user for security
- MicroTeX is automatically built and configured during image creation
- Build artifacts are cached in a Docker volume for faster rebuilds

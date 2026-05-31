# 1. Use the official vLLM image to avoid compiling CUDA kernels
FROM vllm/vllm-openai:latest

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# 2. Install system-level dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# 3. Install toolkit, CJK font support
RUN apt-get install -y x11-apps feh vim tmux curl wget python-is-python3
RUN apt-get install -y fonts-wqy-microhei

# Remove apt cache
RUN rm -rf /var/lib/apt/lists/*

# 4. Install the Vision and Quantization Python libraries
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python \
    autoawq \
    pillow

# 5. Set the working directory inside the container
WORKDIR /app

# Workaround for Qt/X11 shared memory issue in Docker
ENV QT_X11_NO_MITSHM=1

# Default command
CMD ["/bin/bash"]

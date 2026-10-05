FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies, SSH server, curl, and iptables (required for Tailscale)
# Install system dependencies, SSH server, curl, iptables, AND build-essential (for Triton C++ compiling)
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    openssh-server curl iptables \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH so you can log in via VS Code
RUN mkdir /var/run/sshd
# Setting the root password to 'reat2026' (you can change this)
RUN echo 'root:reat2026' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Install Tailscale
RUN curl -fsSL https://tailscale.com/install.sh | sh

# Install PyTorch and Python dependencies
# Pull the absolute latest cutting-edge PyTorch build to support RTX 5090 (sm_120)
RUN pip install --pre --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu132
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install wandb

# Copy project files
COPY . .

# Expose SSH port
EXPOSE 22

# Start SSH daemon and Tailscale daemon in the background, then keep container alive
CMD /usr/sbin/sshd && tailscaled --tun=userspace-networking --state=/var/lib/tailscale/tailscaled.state
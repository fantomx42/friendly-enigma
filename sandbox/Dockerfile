FROM python:3.14-slim

# Install basic tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Keep container running
CMD ["tail", "-f", "/dev/null"]


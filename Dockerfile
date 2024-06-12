FROM python:3.11-slim

ARG VERSION=3.3.2

# Install necessary dependencies, including libgomp1
RUN apt-get update && \
    apt-get install -y --no-install-recommends git unzip libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install -U pip && \
    pip install --no-cache-dir --no-compile 'git+https://github.com/goeckslab/smart-report.git@17df590f3ceb065add099f37b4874c85bd275014' && \
    pip install --no-cache-dir --no-compile pycaret==${VERSION} && \
    pip install --no-cache-dir --no-compile explainerdashboard

# Clean up unnecessary packages
RUN apt-get -y autoremove && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

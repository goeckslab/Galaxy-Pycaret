FROM python:3.11-slim

ARG VERSION=3.3.2

# Install necessary dependencies, including libgomp1
RUN apt-get update && \
    apt-get install -y --no-install-recommends git unzip libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libgdk-pixbuf2.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libpango-1.0-0 \
    libglib2.0-0 \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install -U pip && \
    pip install --no-cache-dir --no-compile joblib && \
    pip install --no-cache-dir --no-compile h5py && \
    pip install --no-cache-dir --no-compile weasyprint && \
    pip install --no-cache-dir --no-compile markdown2 && \
    pip install --no-cache-dir --no-compile pycaret[analysis,models]==${VERSION} && \
    pip install --no-cache-dir --no-compile explainerdashboard

# Clean up unnecessary packages
RUN apt-get -y autoremove && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

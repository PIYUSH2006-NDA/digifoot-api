FROM python:3.11-slim

# System deps for Open3D + OpenCV + ultralytics
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy + install requirements (cached layer)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy app code
COPY --chown=user . .

# Create runtime dirs
RUN mkdir -p scans stls weights outputs validation_set

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
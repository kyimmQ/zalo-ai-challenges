FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY simple-tad/ ./simple-tad/
COPY model.py dataset.py inference.py ./
COPY best_model.pth ./

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "inference.py"]

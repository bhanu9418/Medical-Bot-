

FROM python:3.10-slim-buster

WORKDIR /app

# 1️⃣ Install system dependencies (IMPORTANT)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2️⃣ Copy only requirements first (cache-friendly)
COPY requirements.txt .

# 3️⃣ Upgrade pip & install deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4️⃣ Copy app source
COPY . .

CMD ["python", "app.py"]
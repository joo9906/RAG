FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 (EasyOCR, docling 등에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# pdfplumber 0.11.8 이 pdfminer.six==20251107 을 강제 요구하지만
# mineru 는 >=20251230 을 요구해 충돌함.
# requirements.txt 로 나머지를 먼저 설치해 mineru 가 원하는 버전을 확정한 뒤,
# pdfplumber 를 --no-deps 로 재설치해 pdfminer.six 핀 검사를 우회한다.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps pdfplumber==0.11.8

COPY . .

CMD ["python", "total_process.py"]

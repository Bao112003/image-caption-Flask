# Sử dụng Python 3.10 (tương thích TensorFlow)
FROM python:3.10-slim

# Tắt buffering của Python
ENV PYTHONUNBUFFERED=1

# Tạo thư mục làm việc trong container
WORKDIR /app

# Cài đặt các thư viện hệ thống cần cho TensorFlow/Keras + Pillow + numpy
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements trước để cache install
COPY requirements.txt .

# Cài đặt dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn dự án vào container
COPY . .

# Expose port Flask
EXPOSE 5000

# Lệnh chạy app Flask
CMD ["python", "app.py"]

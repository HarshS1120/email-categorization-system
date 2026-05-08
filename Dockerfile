FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools==68.2.2 wheel==0.41.2 && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Copy app code
COPY . .

# Train model
RUN python main.py

EXPOSE 10000
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]
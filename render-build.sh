#!/bin/bash
echo "=== Starting Render Build Process ==="

# Install system dependencies
apt-get update
apt-get install -y gcc g++ build-essential python3-dev

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Train the model
python main.py

echo "=== Build Complete ==="
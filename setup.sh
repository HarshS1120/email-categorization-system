#!/bin/bash
# setup.sh - Run on Render startup

# Download NLTK data
python -m nltk.downloader stopwords
python -m nltk.downloader punkt

# Create models directory if not exists
mkdir -p models

# Check if model exists, if not train it
if [ ! -f "models/best_model.pkl" ]; then
    echo "Training model..."
    python main.py
fi

echo "Setup complete!"
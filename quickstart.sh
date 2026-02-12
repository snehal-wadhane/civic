#!/bin/bash
# Quick Start Script for CivicLink

echo "================================"
echo "CivicLink v2.0 Quick Start"
echo "================================"
echo

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Python 3.9+ required"; exit 1; }

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env if not exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo
    echo "⚠️  IMPORTANT: Edit .env file with your Supabase credentials!"
    echo "   nano .env"
    echo
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p models logs

# Check for ML model
if [ ! -f models/pothole_yolov8_best.pt ]; then
    echo
    echo "⚠️  WARNING: ML model not found!"
    echo "   Please copy pothole_yolov8_best.pt to models/ directory"
    echo
fi

# Display database setup instructions
echo
echo "================================"
echo "Next Steps:"
echo "================================"
echo
echo "1. Edit .env file with your Supabase credentials:"
echo "   nano .env"
echo
echo "2. Set up database:"
echo "   python setup_database.py"
echo "   Then copy SQL to Supabase SQL Editor and run it"
echo
echo "3. Copy ML model (if not done):"
echo "   cp /path/to/pothole_yolov8_best.pt models/"
echo
echo "4. Start the server:"
echo "   uvicorn app:app --reload"
echo
echo "5. Access API documentation:"
echo "   http://localhost:8000/docs"
echo
echo "================================"
echo "Setup complete!"
echo "================================"

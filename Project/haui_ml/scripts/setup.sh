#!/bin/bash

# Setup Script for Fake Reviews Detection
# This script sets up the entire environment from scratch

echo "🛠️  FAKE REVIEWS DETECTION - ENVIRONMENT SETUP"
echo "=============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $python_version"

# Move to project root directory
cd "$(dirname "$0")/.."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -d "src" ]; then
    echo "❌ Error: Could not find project structure"
    echo "   Make sure 'requirements.txt' and 'src/' exist"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "🔄 Virtual environment already exists, recreating..."
    rm -rf .venv
fi

python3 -m venv .venv
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements"
    exit 1
fi

# Mark as installed
touch .venv/installed

# Download NLTK data
echo "📝 Setting up NLTK data..."
python -c "
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print('📥 Downloading NLTK stopwords...')
nltk.download('stopwords', quiet=True)
print('📥 Downloading NLTK punkt...')
nltk.download('punkt', quiet=True)
print('✅ NLTK data downloaded successfully')
"

# Check model files
echo "🤖 Checking model files..."
if [ ! -d "model_weights" ]; then
    echo "⚠️  Model weights directory not found!"
    echo "   You'll need to train models or copy them to 'model_weights/' directory"
elif [ $(ls model_weights/*.pkl 2>/dev/null | wc -l) -eq 0 ]; then
    echo "⚠️  No model files found in model_weights/"
    echo "   You'll need to train models first"
else
    model_count=$(ls model_weights/*.pkl 2>/dev/null | wc -l)
    echo "✅ Found $model_count model files"
fi

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x test.sh demo.sh start_app.sh run_app.sh 2>/dev/null

# Test the setup
echo ""
echo "🧪 Testing the setup..."
echo "======================"
cd src
python test_app.py
test_result=$?
cd ..

echo ""
echo "✅ SETUP COMPLETED!"
echo "=================="

if [ $test_result -eq 0 ]; then
    echo "🎉 Everything is working correctly!"
    echo ""
    echo "📋 Available commands:"
    echo "  ./test.sh      - Run quick tests"
    echo "  ./demo.sh      - Run interactive demo"
    echo "  ./start_app.sh - Start web application"
    echo "  ./run_app.sh   - Alternative app launcher"
    echo ""
    echo "🚀 Ready to start! Run './start_app.sh' to launch the web interface."
else
    echo "⚠️  Setup completed but some tests failed."
    echo "   This might be due to missing model files."
    echo "   Train your models first, then run './test.sh' to verify."
fi

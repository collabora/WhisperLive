#! /bin/bash

# Detect the operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS, using Homebrew for installation"

    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first: https://brew.sh/"
        exit 1
    fi

    # Install packages using Homebrew
    brew install portaudio wget
else
    # Linux (Debian/Ubuntu)
    echo "Detected Linux, using apt-get for installation"
    apt-get install portaudio19-dev wget -y
fi

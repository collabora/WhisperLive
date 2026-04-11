#!/bin/bash

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
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
    fi

    if [[ "${ID:-}" == "fedora" ]]; then
        echo "Detected Fedora, using dnf for installation"
        dnf install -y portaudio-devel wget
    else
        echo "Detected Linux (assuming Debian/Ubuntu), using apt-get for installation"
        apt-get install -y portaudio19-dev wget
    fi
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi
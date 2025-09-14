#!/usr/bin/env bash
set -euo pipefail

# Setup swap file for Jetson Nano to enable larger Whisper models
# Usage: ./setup_swap.sh [SIZE_GB]
# Default size: 16GB

SIZE_GB="${1:-16}"

echo "=== Jetson Nano Swap Setup ==="
echo "Setting up ${SIZE_GB}GB swap file..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    echo "Error: This script should not be run as root directly."
    echo "It will use sudo for specific commands that require elevated privileges."
    exit 1
fi

# Check if swap already exists
if swapon --show | grep -q "/swapfile"; then
    echo "Warning: Swap file already exists and is active."
    echo "Current swap status:"
    swapon --show
    echo ""
    read -p "Do you want to continue and create additional swap? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Check available disk space
AVAILABLE_SPACE=$(df / | awk 'NR==2 {print int($4/1024/1024)}')  # Available space in GB
echo "Available disk space: ${AVAILABLE_SPACE}GB"

if [[ $SIZE_GB -gt $AVAILABLE_SPACE ]]; then
    echo "Error: Not enough disk space. Requested: ${SIZE_GB}GB, Available: ${AVAILABLE_SPACE}GB"
    echo "Consider using a smaller size or freeing up disk space."
    exit 1
fi

# Create swap file
echo "Creating ${SIZE_GB}GB swap file..."
sudo fallocate -l "${SIZE_GB}G" /swapfile

# Set proper permissions
echo "Setting swap file permissions..."
sudo chmod 600 /swapfile

# Format as swap
echo "Formatting swap file..."
sudo mkswap /swapfile

# Add to fstab for persistence
echo "Adding swap to /etc/fstab for persistence..."
if ! grep -q "/swapfile" /etc/fstab; then
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "Added swap entry to /etc/fstab"
else
    echo "Swap entry already exists in /etc/fstab"
fi

# Enable swap
echo "Enabling swap..."
sudo swapon /swapfile

# Verify swap is active
echo ""
echo "=== Swap Setup Complete ==="
echo "Current swap status:"
swapon --show

echo ""
echo "=== Important Notes ==="
echo "• Swap is now active and will persist across reboots"
echo "• For Whisper models:"
echo "  - medium: requires ≥8GB swap"
echo "  - large-v2: requires ≥16GB swap"
echo "• SD cards will experience wear with heavy swap usage"
echo "• Consider using a USB SSD for swap on long-term deployments"
echo "• Monitor swap usage with: free -h"
echo "• To disable swap: sudo swapoff /swapfile"

echo ""
echo "=== Model Recommendations ==="
if [[ $SIZE_GB -ge 16 ]]; then
    echo "✓ Can run Whisper large-v2 models"
    echo "  Set: export TRANSCRIBE_MODEL=large-v2"
elif [[ $SIZE_GB -ge 8 ]]; then
    echo "✓ Can run Whisper medium models"
    echo "  Set: export TRANSCRIBE_MODEL=medium"
else
    echo "⚠ Limited to smaller models (small, base, tiny)"
    echo "  Consider increasing swap size for better accuracy"
fi

echo ""
echo "=== Usage Example ==="
echo "export TRANSCRIBE_MODEL=medium"
echo "export ALLOW_SWAP=true"
echo "python3 transcriber.py audio.wav"

#!/bin/bash
# Setup script for Whisper-TRT Automation Manager
# This script sets up the automated recording and transcription system as a systemd service.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="whisper-automation"
SERVICE_FILE="${SCRIPT_DIR}/${SERVICE_NAME}.service"
SYSTEMD_DIR="/etc/systemd/system"
CONFIG_FILE="${SCRIPT_DIR}/automation_config.yaml"

echo -e "${BLUE}Whisper-TRT Automation Setup${NC}"
echo "================================"

# Check if running as root for systemd operations
if [[ $EUID -eq 0 ]]; then
    echo -e "${YELLOW}Warning: Running as root. This is needed for systemd service installation.${NC}"
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker first"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    echo "Please start Docker daemon first"
    exit 1
fi

# Check if whisper-trt container image exists
echo -e "${BLUE}Checking Docker container...${NC}"
if ! docker image inspect whisper-trt:latest &> /dev/null; then
    echo -e "${RED}Error: whisper-trt:latest container image not found${NC}"
    echo "Please build the container first:"
    echo "  docker build -t whisper-trt:latest ."
    exit 1
fi

echo -e "${GREEN}Docker container found: whisper-trt:latest${NC}"

# Check if Python 3 is available (minimal requirement for automation manager)
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Python 3 is needed for the automation manager (not for recording/transcription)"
    exit 1
fi

# Create necessary directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p ~/recordings
mkdir -p ~/transcriptions
mkdir -p ~/.cache/whisper_trt/logs

# Make automation script executable
chmod +x "${SCRIPT_DIR}/automation_manager.py"

# Test the automation script (basic syntax check)
echo -e "${BLUE}Testing automation script...${NC}"
# Clean any existing cache files
rm -rf "${SCRIPT_DIR}/__pycache__" 2>/dev/null || true
python3 -c "import ast; ast.parse(open('${SCRIPT_DIR}/automation_manager.py').read())" || {
    echo -e "${RED}Error: Automation script has syntax errors${NC}"
    exit 1
}
echo -e "${GREEN}Automation script syntax OK${NC}"

# Install systemd service
if [[ $EUID -eq 0 ]]; then
    echo -e "${BLUE}Installing systemd service...${NC}"
    
    # Copy service file
    cp "${SERVICE_FILE}" "${SYSTEMD_DIR}/"
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service
    systemctl enable "${SERVICE_NAME}"
    
    echo -e "${GREEN}Service installed and enabled${NC}"
    echo ""
    echo "To start the service:"
    echo "  sudo systemctl start ${SERVICE_NAME}"
    echo ""
    echo "To check status:"
    echo "  sudo systemctl status ${SERVICE_NAME}"
    echo ""
    echo "To view logs:"
    echo "  sudo journalctl -u ${SERVICE_NAME} -f"
    echo ""
    echo "To stop the service:"
    echo "  sudo systemctl stop ${SERVICE_NAME}"
    
else
    echo -e "${YELLOW}Not running as root. Skipping systemd service installation.${NC}"
    echo ""
    echo "To install the systemd service, run:"
    echo "  sudo ${BASH_SOURCE[0]}"
    echo ""
    echo "To run manually:"
    echo "  python3 ${SCRIPT_DIR}/automation_manager.py --config ${CONFIG_FILE}"
fi

# Show configuration
echo -e "${BLUE}Current Configuration:${NC}"
echo "Recording: 6:30 AM - 8:00 PM (10-minute chunks)"
echo "Transcription: 8:00 PM (small model, 3 speakers)"
echo "Container: whisper-trt:latest (Docker)"
echo "Output directories:"
echo "  Recordings: ~/recordings"
echo "  Transcriptions: ~/transcriptions"
echo "  Logs: ~/.cache/whisper_trt/logs"
echo ""

# Show next steps
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Review the configuration file: ${CONFIG_FILE}"
echo "2. Test the system: python3 ${SCRIPT_DIR}/automation_manager.py --status"
echo "3. Start the service: sudo systemctl start ${SERVICE_NAME}"
echo "4. Monitor logs: sudo journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "For manual operation:"
echo "  python3 ${SCRIPT_DIR}/automation_manager.py --config ${CONFIG_FILE}"
echo ""
echo -e "${BLUE}Note:${NC} Recording and transcription run inside the Docker container"
echo "All Whisper-TRT dependencies are available in the container"


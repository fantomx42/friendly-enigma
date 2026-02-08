#!/bin/bash
#===============================================================================
# Void AI Portable Installer
#
# Creates a self-contained Void AI development environment at ~/VoidAI
# Target: CachyOS with AMD Radeon RX 9070 XT (RDNA 4)
#
# Components:
#   - Directory structure for models, scripts, logs, config
#   - 64GB swap file for large model loading
#   - llama.cpp-hip (AMD ROCm support via AUR)
#   - Qwen3-Coder-Next Q4_K_M model (~48GB)
#   - Void Editor IDE
#   - Desktop launcher
#
# Usage: ./install_void_ai_portable.sh
#
# This script is idempotent - safe to run multiple times.
#===============================================================================

set -euo pipefail

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------

VOID_HOME="$HOME/VoidAI"
SWAP_SIZE_GB=64
SWAP_FILE="/swapfile"
MIN_FREE_SPACE_GB=100

# Model configuration
MODEL_NAME="Qwen3-Coder-Next-Q4_K_M"
MODEL_URL_BASE="https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-Q4_K_M"
MODEL_FILES=(
    "Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf"
    "Qwen3-Coder-Next-Q4_K_M-00002-of-00004.gguf"
    "Qwen3-Coder-Next-Q4_K_M-00003-of-00004.gguf"
    "Qwen3-Coder-Next-Q4_K_M-00004-of-00004.gguf"
)

# Void Editor (binaries repo - filenames include version, so we fetch dynamically)
VOID_RELEASES_API="https://api.github.com/repos/voideditor/binaries/releases/latest"

#-------------------------------------------------------------------------------
# Color Definitions
#-------------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

#-------------------------------------------------------------------------------
# Helper Functions
#-------------------------------------------------------------------------------

print_header() {
    echo -e "\n${BOLD}${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║${NC} ${CYAN}$1${NC}"
    echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}\n"
}

print_step() {
    echo -e "${GREEN}▶${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

spinner() {
    local pid=$1
    local message=$2
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0

    while kill -0 "$pid" 2>/dev/null; do
        printf "\r${CYAN}${spin:i++%10:1}${NC} %s" "$message"
        sleep 0.1
    done
    printf "\r"
}

check_command() {
    if command -v "$1" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

get_free_space_gb() {
    local path="$1"
    df -BG "$path" | awk 'NR==2 {gsub(/G/,"",$4); print $4}'
}

get_swap_total_gb() {
    free -g | awk '/Swap:/ {print $2}'
}

#-------------------------------------------------------------------------------
# Pre-flight Checks
#-------------------------------------------------------------------------------

preflight_checks() {
    print_header "Pre-flight Checks"

    local errors=0

    # Check for required commands
    print_step "Checking required commands..."
    for cmd in curl git pacman makepkg; do
        if check_command "$cmd"; then
            print_success "$cmd found"
        else
            print_error "$cmd not found"
            ((errors++))
        fi
    done

    # Check free space
    print_step "Checking disk space..."
    local free_space
    free_space=$(get_free_space_gb "$HOME")
    if (( free_space >= MIN_FREE_SPACE_GB )); then
        print_success "Free space: ${free_space}GB (minimum: ${MIN_FREE_SPACE_GB}GB)"
    else
        print_error "Insufficient free space: ${free_space}GB (need ${MIN_FREE_SPACE_GB}GB)"
        ((errors++))
    fi

    # Check for existing installation
    if [[ -d "$VOID_HOME" ]]; then
        print_warn "Existing installation found at $VOID_HOME"
        echo -e "    Options: [s]kip existing files, [b]ackup and reinstall, [c]ancel"
        read -rp "    Choice [s/b/c]: " choice
        case "${choice,,}" in
            b)
                local backup_dir="${VOID_HOME}.backup.$(date +%Y%m%d_%H%M%S)"
                print_step "Backing up to $backup_dir..."
                mv "$VOID_HOME" "$backup_dir"
                print_success "Backup complete"
                ;;
            c)
                print_info "Installation cancelled"
                exit 0
                ;;
            *)
                print_info "Will skip existing files"
                ;;
        esac
    fi

    if (( errors > 0 )); then
        print_error "Pre-flight checks failed with $errors error(s)"
        exit 1
    fi

    print_success "All pre-flight checks passed"
}

#-------------------------------------------------------------------------------
# Directory Setup
#-------------------------------------------------------------------------------

setup_directories() {
    print_header "Setting Up Directory Structure"

    local dirs=(
        "$VOID_HOME"
        "$VOID_HOME/models"
        "$VOID_HOME/scripts"
        "$VOID_HOME/logs"
        "$VOID_HOME/config"
        "$VOID_HOME/void"
    )

    for dir in "${dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            print_info "Directory exists: $dir"
        else
            mkdir -p "$dir"
            print_success "Created: $dir"
        fi
    done

    # Create a simple icon (base64 encoded minimal PNG)
    if [[ ! -f "$VOID_HOME/icon.png" ]]; then
        print_step "Creating icon..."
        # Simple 64x64 purple gradient icon as base64
        base64 -d > "$VOID_HOME/icon.png" << 'ICON_EOF'
iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEwAACxMBAJqcGAAAA4BJREFUeJztmk1u2zAQhe8iXaRIF+kR6gt0kR6hR+giXfQIOUKP0EW6
SBfpIl2ki3QRj0hKpChRki3Zdh7gQKZE/nwznBmOZPx4fAQwv7+9AwC8vL4DAL6+PD8BAN6/vQMA
fHt9AwB8e3kFAHx7ewUA+Pr2BgD48vYOAPDl/R0A8OntAwDw6f0DAPD+8REA8O7jIwDg3cdHAMDb
j48AgDcfHwEArz4+AgBefXwEALz8+AgAePHxEQDw/OMjAODZx0cAwNOPjwCAJx8fAQCPPz4CAB59
fAQAPPz4CAB48PERAPH4CADR+AgA2I2PAIDd+AgA2ImPAICd+AgA2I6PAICt+AgA2IqPAIAt+wgA
2LKPAIB1+wgAWLePAIA1+wgAWLOPAIA1+wgAWLWPAIBV+wgAWLGPAIAV+wgAWLaPAIBl+wgAWLKP
AIAl+wgAWLSPAIBF+wgAWLCPAIAF+wgAmLePAIB5+wgAmLOPAIA5+wgAmLWPAIBZ+wgAmLGPAIAZ
+wgAmLaPAIBp+wgAmLKPAIAp+wgAmLSPAIBJ+wgAmLCPAIAJ+wgAGLePAIBx+wgAGLOPAIAx+wgA
GLWPAIBRvAcA/AsA+BcA8C8A4F8AwL8AgH8BAP8CAP4FAPwLAPgXAPAvAOBfAMC/AIB/AQD/AgD+
BQD8CwD4FwDwLwDgXwDAvwCAfwEA/wIA/gUA/AsA+BcA8C8A4F8AwL8AgH8BAP8CAP4FAPwLAPgX
APAvAOBfAMC/AIB/AQD/AgD+BQD8CwD4FwDwLwDgXwDAvwCAfwEA/wIA/gUA/AsA+BcA8C8A4F8A
wL8AgH8BAP8CAP4FAPwLAPgXAPAvAOBfAMC/AIB/AQD/AgD+BQD8CwD4FwDwLwDgXwDAvwCAfwEA
/wIA/gUA/AsA+BcA8C8A4F8AwL8AgH8BAP8CAP4FAPwLAPgXAPAvAOBfAMC/AIB/AQD/AgD+BQD8
CwD4FwDwLwDgXwDAvwCAfwEA/wIA/gUA/AsA+BcA8C8A4F8AwL8AgH8BAP8CAP4FAPwLAPgXAPAv
AOBfAMC/AIB/AQD/AgD+BQD8CwD4FwDwLwDgXwDAvwCAfwEA/wIA/gUA/AsA+BcA8C8A4F8AwL8A
gH8BAP8CAP4FAPwLAPgXAPAvAOBfAMC/AIB/AQD/AgD+BQD8CwD4FwDwLwDgXwDAvwCAfwEA/wIA
/gX+A7n7D5GpSiONAAAAAElFTkSuQmCC
ICON_EOF
        print_success "Created icon.png"
    fi
}

#-------------------------------------------------------------------------------
# Swap File Setup
#-------------------------------------------------------------------------------

setup_swap() {
    print_header "Configuring Swap File"

    local current_swap
    current_swap=$(get_swap_total_gb)

    print_info "Current swap: ${current_swap}GB"

    if (( current_swap >= 32 )); then
        print_success "Sufficient swap already configured (${current_swap}GB)"
        return 0
    fi

    print_warn "Insufficient swap for large model loading"
    print_step "Creating ${SWAP_SIZE_GB}GB swap file (requires sudo)..."

    if [[ -f "$SWAP_FILE" ]]; then
        print_info "Swap file exists, checking size..."
        local existing_size
        existing_size=$(stat -c%s "$SWAP_FILE" 2>/dev/null || echo 0)
        local target_size=$((SWAP_SIZE_GB * 1024 * 1024 * 1024))

        if (( existing_size >= target_size )); then
            print_success "Existing swap file is adequate"
        else
            print_step "Resizing swap file..."
            sudo swapoff "$SWAP_FILE" 2>/dev/null || true
            sudo rm -f "$SWAP_FILE"
        fi
    fi

    if [[ ! -f "$SWAP_FILE" ]]; then
        print_step "Allocating ${SWAP_SIZE_GB}GB swap file..."

        # Try fallocate first, fall back to dd
        if ! sudo fallocate -l "${SWAP_SIZE_GB}G" "$SWAP_FILE" 2>/dev/null; then
            print_warn "fallocate failed, using dd (slower)..."
            sudo dd if=/dev/zero of="$SWAP_FILE" bs=1G count="$SWAP_SIZE_GB" status=progress
        fi

        print_step "Setting permissions..."
        sudo chmod 600 "$SWAP_FILE"

        print_step "Formatting swap..."
        sudo mkswap "$SWAP_FILE"
    fi

    # Enable swap if not already active
    if ! swapon --show | grep -q "$SWAP_FILE"; then
        print_step "Enabling swap..."
        sudo swapon "$SWAP_FILE"
    fi

    # Add to fstab if not present
    if ! grep -q "$SWAP_FILE" /etc/fstab; then
        print_step "Adding swap to /etc/fstab..."
        echo "$SWAP_FILE none swap defaults 0 0" | sudo tee -a /etc/fstab >/dev/null
    fi

    # Configure swappiness
    print_step "Configuring VM parameters..."
    sudo tee /etc/sysctl.d/99-void-ai.conf >/dev/null << 'SYSCTL_EOF'
# Void AI swap configuration
# Moderate swappiness for large model loading
vm.swappiness=60
# Reduce cache pressure to keep model pages resident
vm.vfs_cache_pressure=50
SYSCTL_EOF

    sudo sysctl -p /etc/sysctl.d/99-void-ai.conf >/dev/null 2>&1 || true

    print_success "Swap configured: $(swapon --show | grep -v NAME)"
}

#-------------------------------------------------------------------------------
# llama.cpp Installation (AUR)
#-------------------------------------------------------------------------------

install_llama_cpp() {
    print_header "Installing llama.cpp (HIP/ROCm)"

    if check_command llama-server; then
        print_success "llama-server already installed"
        llama-server --version 2>/dev/null || true
        return 0
    fi

    # Check for yay or paru
    local aur_helper=""
    if check_command yay; then
        aur_helper="yay"
    elif check_command paru; then
        aur_helper="paru"
    fi

    if [[ -n "$aur_helper" ]]; then
        print_step "Installing llama.cpp-hip via $aur_helper..."
        $aur_helper -S --needed --noconfirm llama.cpp-hip
    else
        print_step "No AUR helper found, building manually..."

        local build_dir="/tmp/llama-cpp-hip-build"
        rm -rf "$build_dir"
        mkdir -p "$build_dir"
        cd "$build_dir"

        print_step "Cloning AUR package..."
        git clone https://aur.archlinux.org/llama.cpp-hip.git .

        print_step "Building package (this may take a while)..."
        makepkg -si --noconfirm

        cd - >/dev/null
        rm -rf "$build_dir"
    fi

    if check_command llama-server; then
        print_success "llama.cpp installed successfully"
    else
        print_error "llama-server not found after installation"
        print_info "You may need to install llama.cpp manually"
        return 1
    fi
}

#-------------------------------------------------------------------------------
# Model Download
#-------------------------------------------------------------------------------

download_model() {
    print_header "Downloading Qwen3-Coder Model"

    local model_dir="$VOID_HOME/models"
    local total_files=${#MODEL_FILES[@]}
    local current=0

    for model_file in "${MODEL_FILES[@]}"; do
        ((++current))
        local filepath="$model_dir/$model_file"

        print_step "[$current/$total_files] $model_file"

        # Skip if file exists and is reasonably large (>10GB = likely complete)
        if [[ -f "$filepath" ]]; then
            local size
            size=$(stat -c%s "$filepath" 2>/dev/null || echo 0)
            if (( size > 3221225472 )); then  # >3GB = likely complete
                print_info "  Already downloaded ($(numfmt --to=iec $size))"
                continue
            else
                print_warn "  Incomplete download detected, resuming..."
            fi
        fi

        # Download with resume support
        curl -L -C - --progress-bar \
            -o "$filepath" \
            "${MODEL_URL_BASE}/${model_file}"

        if [[ -f "$filepath" ]]; then
            local final_size
            final_size=$(stat -c%s "$filepath")
            print_success "  Downloaded ($(numfmt --to=iec $final_size))"
        else
            print_error "  Download failed"
            return 1
        fi
    done

    print_success "All model files downloaded"
}

#-------------------------------------------------------------------------------
# Void Editor Installation
#-------------------------------------------------------------------------------

install_void_editor() {
    print_header "Installing Void Editor"

    local void_dir="$VOID_HOME/void"
    local void_exe="$void_dir/void"

    if [[ -x "$void_exe" ]]; then
        print_success "Void Editor already installed"
        return 0
    fi

    print_step "Fetching latest release info..."

    # Get the download URL for linux-x64 tar.gz from GitHub API
    local download_url
    download_url=$(curl -s "$VOID_RELEASES_API" | grep -oP '"browser_download_url":\s*"\K[^"]+linux-x64[^"]+\.tar\.gz' | head -1)

    if [[ -z "$download_url" ]]; then
        print_error "Could not find linux-x64 download URL"
        print_info "You can manually download from: https://github.com/voideditor/binaries/releases"
        return 1
    fi

    print_step "Downloading Void Editor..."
    print_info "URL: $download_url"

    local download_file="/tmp/void-linux.tar.gz"

    if curl -L --progress-bar -o "$download_file" "$download_url"; then
        print_step "Extracting..."
        tar -xzf "$download_file" -C "$void_dir" --strip-components=1
        rm -f "$download_file"

        # Make executable
        chmod +x "$void_dir/void" 2>/dev/null || true
        chmod +x "$void_dir/Void" 2>/dev/null || true

        # Find the actual executable
        if [[ -x "$void_dir/void" ]]; then
            print_success "Void Editor installed"
        elif [[ -x "$void_dir/Void" ]]; then
            ln -sf "$void_dir/Void" "$void_dir/void"
            print_success "Void Editor installed (as Void)"
        else
            print_warn "Void executable not found, may need manual setup"
        fi
    else
        print_error "Failed to download Void Editor"
        print_info "You can manually download from: https://github.com/voideditor/binaries/releases"
        return 1
    fi
}

#-------------------------------------------------------------------------------
# Ralph AI Extension
#-------------------------------------------------------------------------------

install_ralph_extension() {
    print_header "Installing Ralph AI Extension"

    # Find the ralph-vscode directory relative to this script
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local ext_dir="$script_dir/ralph-vscode"

    if [[ ! -f "$ext_dir/package.json" ]]; then
        print_warn "ralph-vscode/ not found at $ext_dir, skipping extension install"
        return 0
    fi

    # Check for npm
    if ! check_command npm; then
        print_warn "npm not found, skipping extension build"
        return 0
    fi

    print_step "Installing extension dependencies..."
    (cd "$ext_dir" && npm install --no-audit --no-fund) || {
        print_warn "npm install failed, skipping extension"
        return 0
    }

    print_step "Compiling extension..."
    (cd "$ext_dir" && npx tsc -p ./) || {
        print_warn "TypeScript compilation failed, skipping extension"
        return 0
    }

    print_step "Packaging extension..."
    (cd "$ext_dir" && npx vsce package --no-dependencies 2>/dev/null) || {
        print_warn "VSIX packaging failed, installing from source instead"
    }

    # Install into Void Editor
    local void_exe="$VOID_HOME/void/void"
    [[ ! -x "$void_exe" ]] && void_exe="$VOID_HOME/void/Void"

    local vsix_file
    vsix_file=$(ls "$ext_dir"/*.vsix 2>/dev/null | head -1)

    if [[ -x "$void_exe" ]]; then
        if [[ -n "$vsix_file" ]]; then
            print_step "Installing extension into Void Editor..."
            "$void_exe" --install-extension "$vsix_file" --ozone-platform=x11 2>/dev/null || true
            print_success "Ralph AI extension installed"
        else
            print_info "No .vsix found; use --extensionDevelopmentPath=$ext_dir for dev mode"
        fi
    else
        print_info "Void Editor not found; extension built but not installed"
    fi
}

#-------------------------------------------------------------------------------
# Launcher Script
#-------------------------------------------------------------------------------

create_launcher() {
    print_header "Creating Launcher Script"

    local launcher="$VOID_HOME/scripts/launch.sh"
    local model_file="${MODEL_FILES[0]}"

    cat > "$launcher" << 'LAUNCHER_EOF'
#!/bin/bash
#===============================================================================
# Void AI Launcher
# Starts llama-server with Qwen3-Coder and launches Void Editor
#===============================================================================

set -euo pipefail

# AMD RDNA 4 environment
export HSA_OVERRIDE_GFX_VERSION=12.0.0
export PYTORCH_ROCM_ARCH="gfx1200"

VOID_HOME="$HOME/VoidAI"
MODEL="$VOID_HOME/models/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf"
LOG_FILE="$VOID_HOME/logs/server.log"
PORT=8000

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo -e "${GREEN}Goodbye!${NC}"
}

trap cleanup EXIT INT TERM

# Check if already running
if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo -e "${YELLOW}Server already running on port $PORT${NC}"
else
    echo -e "${CYAN}Starting llama-server...${NC}"

    llama-server \
        --model "$MODEL" \
        --mmap \
        -ngl 18 \
        --ctx-size 8192 \
        --cache-type-k q8_0 \
        --cache-type-v q8_0 \
        --threads 16 \
        --port "$PORT" \
        --log-file "$LOG_FILE" &
    SERVER_PID=$!

    echo -e "${CYAN}Loading Brain...${NC}"

    # Wait for server to be ready (with timeout)
    timeout=300
    elapsed=0
    while ! curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        if (( elapsed >= timeout )); then
            echo -e "${YELLOW}Server startup timeout${NC}"
            exit 1
        fi
        sleep 2
        ((elapsed += 2))
        printf "."
    done
    echo ""

    echo -e "${GREEN}Server ready on port $PORT${NC}"
fi

# Launch Void Editor
if [[ -x "$VOID_HOME/void/void" ]]; then
    echo -e "${CYAN}Launching Void Editor...${NC}"
    "$VOID_HOME/void/void" --ozone-platform=x11 &
elif [[ -x "$VOID_HOME/void/Void" ]]; then
    echo -e "${CYAN}Launching Void Editor...${NC}"
    "$VOID_HOME/void/Void" --ozone-platform=x11 &
else
    echo -e "${YELLOW}Void Editor not found, server running at http://127.0.0.1:$PORT${NC}"
fi

# Keep script running to maintain server
if [[ -n "${SERVER_PID:-}" ]]; then
    wait "$SERVER_PID"
fi
LAUNCHER_EOF

    chmod +x "$launcher"
    print_success "Created $launcher"
}

#-------------------------------------------------------------------------------
# Desktop Entry
#-------------------------------------------------------------------------------

create_desktop_entry() {
    print_header "Creating Desktop Entry"

    local desktop_dir="$HOME/.local/share/applications"
    local desktop_file="$desktop_dir/void-ai.desktop"

    mkdir -p "$desktop_dir"

    cat > "$desktop_file" << DESKTOP_EOF
[Desktop Entry]
Name=Void AI
Comment=Void Editor with local Qwen3-Coder AI
Exec=$VOID_HOME/scripts/launch.sh
Icon=$VOID_HOME/icon.png
Type=Application
Categories=Development;IDE;
Terminal=true
StartupNotify=true
Keywords=void;ai;llm;code;editor;
DESKTOP_EOF

    chmod +x "$desktop_file"

    # Update desktop database
    update-desktop-database "$desktop_dir" 2>/dev/null || true

    print_success "Created desktop entry: Void AI"
    print_info "You can now launch from your application menu"
}

#-------------------------------------------------------------------------------
# README Generation
#-------------------------------------------------------------------------------

create_readme() {
    print_header "Creating Documentation"

    cat > "$VOID_HOME/README_SETUP.txt" << 'README_EOF'
===============================================================================
                            VOID AI SETUP GUIDE
===============================================================================

QUICK START
-----------
1. Launch from application menu: "Void AI"
   OR run: ~/VoidAI/scripts/launch.sh

2. Wait for "Server ready" message (model loading takes 1-2 minutes)

3. Void Editor will open automatically

-------------------------------------------------------------------------------

VOID EDITOR CONFIGURATION
-------------------------
Open Void settings (Ctrl+,) and configure:

1. Set API endpoint:
   - Provider: OpenAI Compatible
   - Base URL: http://127.0.0.1:8000/v1
   - API Key: (leave empty or use "none")

2. Set model:
   - Model: Qwen3-Coder-Next-Q4_K_M

3. Optional system prompt (Settings > AI > System Prompt):

   You are Qwen3-Coder-Next, an 80B MoE coding assistant (3B active) running locally.
   You excel at code generation, debugging, refactoring, and explanation.
   Provide concise, accurate responses. Use markdown for code blocks.

-------------------------------------------------------------------------------

DIRECTORY STRUCTURE
-------------------
~/VoidAI/
├── models/     # GGUF model files (~48GB)
├── scripts/    # launch.sh
├── logs/       # server.log
├── config/     # settings
├── void/       # Void Editor
├── icon.png    # Application icon
└── README_SETUP.txt

-------------------------------------------------------------------------------

TROUBLESHOOTING
---------------

Server won't start:
  - Check logs: tail -f ~/VoidAI/logs/server.log
  - Verify model files exist: ls -la ~/VoidAI/models/
  - Check GPU: rocm-smi

Out of memory:
  - Verify swap: swapon --show (should show 64GB)
  - Reduce context: edit launch.sh, lower --ctx-size
  - Reduce GPU layers: edit launch.sh, lower -ngl

Slow performance:
  - Check GPU utilization: rocm-smi
  - Ensure HSA_OVERRIDE_GFX_VERSION=12.0.0 is set

Void Editor issues:
  - Reinstall: rm -rf ~/VoidAI/void && run installer again
  - Check permissions: chmod +x ~/VoidAI/void/void

-------------------------------------------------------------------------------

MANUAL SERVER START
-------------------
If you want to run the server separately:

export HSA_OVERRIDE_GFX_VERSION=12.0.0
llama-server \
  --model ~/VoidAI/models/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf \
  --mmap \
  -ngl 18 \
  --ctx-size 8192 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --threads 16 \
  --port 8000

Then use any OpenAI-compatible client with http://127.0.0.1:8000/v1

-------------------------------------------------------------------------------

SYSTEM REQUIREMENTS
-------------------
- GPU: AMD Radeon RX 9070 XT (16GB VRAM) - RDNA 4
- RAM: 32GB DDR5
- Swap: 64GB (configured by installer)
- Storage: ~100GB for model and tools

===============================================================================
README_EOF

    print_success "Created README_SETUP.txt"
}

#-------------------------------------------------------------------------------
# Main Installation Flow
#-------------------------------------------------------------------------------

main() {
    echo -e "${BOLD}${CYAN}"
    cat << 'BANNER'
 __      __   _     _      _    ___
 \ \    / /__(_)__| |    /_\  |_ _|
  \ \/\/ / _ \ / _` |   / _ \  | |
   \_/\_/\___/_\__,_|  /_/ \_\|___|

   Portable Installation Script
BANNER
    echo -e "${NC}"

    print_info "Target directory: $VOID_HOME"
    print_info "Model: $MODEL_NAME (~48GB)"
    print_info "Swap: ${SWAP_SIZE_GB}GB"
    echo ""

    read -rp "Press Enter to begin installation (Ctrl+C to cancel)..."

    # Run installation steps
    preflight_checks
    setup_directories
    setup_swap
    install_llama_cpp
    download_model
    install_void_editor
    install_ralph_extension
    create_launcher
    create_desktop_entry
    create_readme

    # Final summary
    print_header "Installation Complete"

    echo -e "${GREEN}Void AI has been installed to: $VOID_HOME${NC}"
    echo ""
    echo "To start:"
    echo "  1. From application menu: Look for 'Void AI'"
    echo "  2. From terminal: ~/VoidAI/scripts/launch.sh"
    echo ""
    echo "Documentation: ~/VoidAI/README_SETUP.txt"
    echo ""

    print_success "Installation successful!"
}

# Run main function
main "$@"

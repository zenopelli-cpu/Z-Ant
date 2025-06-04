#!/usr/bin/env bash

# Zig Environment Sourcing Script
# Usage: source ./source_zig.sh [version]

ZIG_VERSION=${1:-"0.13.0"}
ZIG_BASE_DIR="$HOME/.zig"
ZIG_INSTALL_DIR="$ZIG_BASE_DIR/zig-$ZIG_VERSION"
ZIG_CACHE_DIR="$HOME/.cache/zig"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Zig environment...${NC}"

# Create directories
mkdir -p "$ZIG_BASE_DIR"
mkdir -p "$ZIG_CACHE_DIR"

# Function to download and install Zig
install_zig() {
    local version=$1
    local arch=$(uname -m)
    local os="linux"
    
    case $arch in
        x86_64) arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *) echo -e "${RED}Unsupported architecture: $arch${NC}"; return 1 ;;
    esac
    
    local zig_url="https://ziglang.org/download/$version/zig-$os-$arch-$version.tar.xz"
    local zig_tar="$ZIG_BASE_DIR/zig-$version.tar.xz"
    
    echo -e "${YELLOW}Downloading Zig $version...${NC}"
    if ! curl -L "$zig_url" -o "$zig_tar"; then
        echo -e "${RED}Failed to download Zig $version${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Extracting Zig $version...${NC}"
    if ! tar -xf "$zig_tar" -C "$ZIG_BASE_DIR"; then
        echo -e "${RED}Failed to extract Zig $version${NC}"
        return 1
    fi
    
    # Rename extracted directory
    local extracted_dir=$(tar -tf "$zig_tar" | head -1 | cut -f1 -d"/")
    mv "$ZIG_BASE_DIR/$extracted_dir" "$ZIG_INSTALL_DIR"
    
    rm "$zig_tar"
    echo -e "${GREEN}Zig $version installed successfully${NC}"
}

# Check if Zig is already installed
if [ ! -f "$ZIG_INSTALL_DIR/zig" ]; then
    echo -e "${YELLOW}Zig $ZIG_VERSION not found, installing...${NC}"
    install_zig "$ZIG_VERSION"
fi

# Add to PATH
export PATH="$ZIG_INSTALL_DIR:$PATH"
export ZIG_GLOBAL_CACHE_DIR="$ZIG_CACHE_DIR"

# Verify installation
if command -v zig >/dev/null 2>&1; then
    ZIG_ACTUAL_VERSION=$(zig version)
    echo -e "${GREEN}✓ Zig $ZIG_ACTUAL_VERSION is now active${NC}"
    echo -e "${GREEN}✓ Zig path: $(which zig)${NC}"
    echo -e "${GREEN}✓ Cache dir: $ZIG_GLOBAL_CACHE_DIR${NC}"
else
    echo -e "${RED}✗ Failed to setup Zig environment${NC}"
    return 1
fi

# Create/update .zigrc for future shells
cat > "$HOME/.zigrc" << EOF
# Zig Environment Configuration
export PATH="$ZIG_INSTALL_DIR:\$PATH"
export ZIG_GLOBAL_CACHE_DIR="$ZIG_CACHE_DIR"
EOF

echo -e "${YELLOW}Tip: Add 'source ~/.zigrc' to your shell profile for persistent setup${NC}" 
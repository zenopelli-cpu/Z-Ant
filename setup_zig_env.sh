#!/usr/bin/env bash

# Zig Environment Setup Script (no installation)
# Usage: source ./setup_zig_env.sh

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Configuring Zig environment...${NC}"

# Find existing Zig installation
ZIG_PATH=""

# Check if zig command exists and is actually Zig (not snap help)
if command -v zig >/dev/null 2>&1; then
    # Test if it's really Zig by checking version output
    if zig version >/dev/null 2>&1; then
        ZIG_PATH=$(which zig)
    fi
fi

# If not found or snap is interfering, try direct paths
if [ -z "$ZIG_PATH" ]; then
    # Try snap-specific paths first
    for path in /snap/zig/current/zig /var/lib/snapd/snap/bin/zig /snap/bin/zig; do
        if [ -f "$path" ] && "$path" version >/dev/null 2>&1; then
            ZIG_PATH="$path"
            break
        fi
    done
    
    # Try other common paths
    if [ -z "$ZIG_PATH" ]; then
        for path in /usr/bin/zig /usr/local/bin/zig ~/.local/bin/zig ~/zig/zig; do
            if [ -f "$path" ] && "$path" version >/dev/null 2>&1; then
                ZIG_PATH="$path"
                break
            fi
        done
    fi
fi

if [ -z "$ZIG_PATH" ]; then
    echo -e "${RED}✗ Working Zig installation not found${NC}"
    echo -e "${YELLOW}Try: snap install zig --classic${NC}"
    return 1
fi

# Get Zig directory and version
ZIG_DIR=$(dirname "$ZIG_PATH")
ZIG_VERSION=$("$ZIG_PATH" version 2>/dev/null || echo "unknown")

# Setup cache directory
ZIG_CACHE_DIR="$HOME/.cache/zig"
mkdir -p "$ZIG_CACHE_DIR"

# Create a wrapper if needed (for snap installations)
ZIG_WRAPPER="$HOME/.local/bin/zig"
mkdir -p "$(dirname "$ZIG_WRAPPER")"

if [[ "$ZIG_PATH" == *"/snap/"* ]]; then
    # Create wrapper for snap installation
    cat > "$ZIG_WRAPPER" << EOF
#!/bin/bash
exec "$ZIG_PATH" "\$@"
EOF
    chmod +x "$ZIG_WRAPPER"
    
    # Add ~/.local/bin to PATH if not already there
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    ZIG_FINAL_PATH="$ZIG_WRAPPER"
else
    # Use direct path
    export PATH="$ZIG_DIR:$PATH"
    ZIG_FINAL_PATH="$ZIG_PATH"
fi

# Export environment variables
export ZIG_GLOBAL_CACHE_DIR="$ZIG_CACHE_DIR"
export ZIG_OPTIMIZE="ReleaseFast"

echo -e "${GREEN}✓ Zig $ZIG_VERSION configured${NC}"
echo -e "${GREEN}✓ Binary: $ZIG_PATH${NC}"
echo -e "${GREEN}✓ Wrapper: $ZIG_FINAL_PATH${NC}"
echo -e "${GREEN}✓ Cache: $ZIG_CACHE_DIR${NC}"

# Test the setup
if "$ZIG_FINAL_PATH" version >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Zig command working${NC}"
else
    echo -e "${RED}✗ Zig command test failed${NC}"
    return 1
fi

# Create .zigrc for persistence
cat > "$HOME/.zigrc" << EOF
# Zig Environment Configuration
export PATH="\$HOME/.local/bin:\$PATH"
export ZIG_GLOBAL_CACHE_DIR="$ZIG_CACHE_DIR"
export ZIG_OPTIMIZE="ReleaseFast"
EOF

echo -e "${YELLOW}Environment configured. Add 'source ~/.zigrc' to ~/.zshrc for persistence${NC}" 
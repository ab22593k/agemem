#!/bin/bash
set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$HOME/.config/opencode"
PLUGIN_DIR="$CONFIG_DIR/plugin"
CONFIG_FILE="$CONFIG_DIR/opencode.json"

install() {
    echo "🚀 Installing AgeMem Global..."

    # 1. Check dependencies
    echo "🔍 Checking dependencies..."
    command -v uv >/dev/null 2>&1 || { echo >&2 "❌ 'uv' is required but not installed. Aborting."; exit 1; }
    command -v docker >/dev/null 2>&1 || { echo >&2 "❌ 'docker' is required but not installed. Aborting."; exit 1; }

    # 2. Sync Python environment
    echo "📦 Syncing Python environment with uv..."
    cd "$PROJECT_ROOT"
    uv sync

    # 3. Check Weaviate Status
    echo "🐳 Checking Weaviate container..."
    if ! docker ps --filter "name=agemem-weaviate-1" --format "{{.Status}}" | grep -q "Up"; then
        echo "⚠️ Weaviate not running. Attempting to start..."
        docker compose up -d weaviate
        sleep 3
    else
        echo "✅ Weaviate is running."
    fi

    # 4. Deploy Global Plugin
    echo "🧩 Deploying global plugin..."
    mkdir -p "$PLUGIN_DIR"
    
    # Inject absolute path into the plugin template
    sed "s|__AGEMEM_PATH__|$PROJECT_ROOT|g" "$PROJECT_ROOT/plugins/agemem.ts" > "$PLUGIN_DIR/agemem.ts"

    # 5. Register MCP Server and Plugin in opencode.json
    echo "⚙️ Updating OpenCode configuration..."
    if [ ! -f "$CONFIG_FILE" ]; then
        echo '{"$schema": "https://opencode.ai/config.json", "plugin": [], "mcp": {}}' > "$CONFIG_FILE"
    fi

    # Use Python to safely update JSON
    python3 <<EOF
import json
import os

config_path = os.path.expanduser("$CONFIG_FILE")
with open(config_path, 'r') as f:
    config = json.load(f)

# Ensure sections exist
if "mcp" not in config: config["mcp"] = {}
if "plugin" not in config: config["plugin"] = []

# Register MCP
config["mcp"]["agemem"] = {
    "type": "local",
    "command": ["python3", "-m", "src.main"],
    "environment": {
        "PYTHONPATH": "$PROJECT_ROOT"
    },
    "enabled": True
}

# Register Plugin
if "agemem" not in config["plugin"]:
    config["plugin"].append("agemem")

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
EOF

    echo "✨ Installation complete! Please restart OpenCode."
}

remove() {
    echo "🗑️ Removing AgeMem Global..."

    # 1. Remove Plugin File
    if [ -f "$PLUGIN_DIR/agemem.ts" ]; then
        rm "$PLUGIN_DIR/agemem.ts"
        echo "✅ Removed plugin file."
    fi

    # 2. Remove from opencode.json
    if [ -f "$CONFIG_FILE" ]; then
        python3 <<EOF
import json
import os

config_path = os.path.expanduser("$CONFIG_FILE")
with open(config_path, 'r+') as f:
    config = json.load(f)
    
    # Remove MCP
    if "mcp" in config and "agemem" in config["mcp"]:
        del config["mcp"]["agemem"]
        print("✅ Removed MCP entry.")
    
    # Remove Plugin
    if "plugin" in config and "agemem" in config["plugin"]:
        config["plugin"].remove("agemem")
        print("✅ Removed Plugin entry.")
    
    f.seek(0)
    json.dump(config, f, indent=2)
    f.truncate()
EOF
    fi

    echo "✨ Removal complete! Please restart OpenCode."
}

case "$1" in
    install)
        install
        ;;
    remove)
        remove
        ;;
    *)
        echo "Usage: $0 {install|remove}"
        exit 1
        ;;
esac

#!/usr/bin/env bash
# One-time setup: install the pacman hook that auto-rebuilds HIP kernels after ROCm updates.
# Run with sudo:  sudo bash scripts/install_hip_hook.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OWNER="$(stat -c '%U' "$REPO_DIR")"

# --- 1. Wrapper at a path with no spaces (pacman Exec can't handle spaces) ---
cat > /usr/local/bin/wheeler-build-hip << EOF
#!/usr/bin/env bash
exec runuser -u $OWNER -- bash "$REPO_DIR/scripts/build_hip.sh" "\$@"
EOF
chmod +x /usr/local/bin/wheeler-build-hip
echo "Installed /usr/local/bin/wheeler-build-hip"

# --- 2. Pacman hook ---
mkdir -p /etc/pacman.d/hooks
cat > /etc/pacman.d/hooks/wheeler-hip.hook << 'EOF'
[Trigger]
Operation = Upgrade
Operation = Install
Type = Package
Target = rocm-hip-sdk
Target = hip-runtime-amd
Target = hip-dev
Target = rocm-opencl-runtime
Target = rocm-hip-libraries

[Action]
Description = Rebuilding Wheeler Memory HIP kernels...
When = PostTransaction
Exec = /usr/local/bin/wheeler-build-hip
EOF
echo "Installed /etc/pacman.d/hooks/wheeler-hip.hook"

echo ""
echo "Done! Next cachyosupdate that touches ROCm will auto-rebuild the .so files."
echo "Manual rebuild: bash \"$REPO_DIR/scripts/build_hip.sh\""
echo "Test trigger:   sudo pacman -S --needed rocm-hip-sdk"

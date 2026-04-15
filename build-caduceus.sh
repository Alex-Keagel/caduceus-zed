#!/bin/bash
# Build Caduceus IDE (Zed fork) from source
# Requires: Xcode (full, not just Command Line Tools), cmake, Rust
set -e

echo "🐍 Building Caduceus IDE..."

# Check prerequisites
if ! xcrun --find metal &>/dev/null; then
    echo "❌ Metal compiler not found. Install Xcode from App Store, then run:"
    echo "   sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer"
    exit 1
fi

if ! command -v cmake &>/dev/null; then
    echo "Installing cmake..."
    brew install cmake
fi

cd "$(dirname "$0")"

# Ensure on caduceus branch
git checkout caduceus-main 2>/dev/null || true

# Build release
echo "⏳ Building (this takes 5-10 minutes)..."
cargo build --release -p zed

# Bundle
echo "📦 Creating app bundle..."
BINARY="target/release/zed"
if [ -f "$BINARY" ]; then
    APP_DIR="target/release/Caduceus.app"
    mkdir -p "$APP_DIR/Contents/MacOS"
    mkdir -p "$APP_DIR/Contents/Resources"
    cp "$BINARY" "$APP_DIR/Contents/MacOS/Caduceus"
    
    cat > "$APP_DIR/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Caduceus</string>
    <key>CFBundleIdentifier</key>
    <string>com.caduceus.ide</string>
    <key>CFBundleVersion</key>
    <string>0.2.0</string>
    <key>CFBundleExecutable</key>
    <string>Caduceus</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15.7</string>
</dict>
</plist>
PLIST
    
    echo "✅ Built: $APP_DIR"
    echo "   Run: open $APP_DIR"
else
    echo "✅ Built: $BINARY"
    echo "   Run: $BINARY"
fi

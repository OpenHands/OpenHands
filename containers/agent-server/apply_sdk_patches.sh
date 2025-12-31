#!/bin/bash
# Script to apply SDK patches at runtime
# This fixes KeyError: 'content' and Missing required parameters: security_risk

set -e

echo "🔧 Applying SDK patches..."

# Wait a bit for SDK to be available (it's installed after agent-server starts)
sleep 5

# Find the SDK installation directory using Python
# Try multiple methods and locations (including PyInstaller _MEIPASS)
SDK_DIR=$(python3 << 'PYEOF'
import sys
import importlib.util
import os
import time
import glob

# Try multiple times with delays
for attempt in range(15):
    try:
        # Method 1: Try PyInstaller _MEIPASS (most likely location)
        meipass_dirs = glob.glob('/tmp/_MEI*')
        for meipass_dir in meipass_dirs:
            sdk_path = os.path.join(meipass_dir, 'openhands', 'sdk')
            if os.path.exists(sdk_path):
                print(sdk_path)
                sys.exit(0)
        
        # Method 2: Try to import and get the file path
        spec = importlib.util.find_spec('openhands.sdk.llm.mixins.fn_call_converter')
        if spec and spec.origin:
            file_path = spec.origin
            # Get the SDK directory (4 levels up from the file)
            sdk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
            if os.path.exists(sdk_dir):
                print(sdk_dir)
                sys.exit(0)
        
        # Method 3: Try site-packages
        import site
        site_packages = site.getsitepackages()
        for sp in site_packages:
            sdk_path = os.path.join(sp, 'openhands', 'sdk')
            if os.path.exists(sdk_path):
                print(sdk_path)
                sys.exit(0)
        
        # Method 4: Try common locations
        common_paths = [
            '/app',
            '/openhands',
            '/usr/local/lib/python3.12/site-packages',
            '/usr/lib/python3.12/site-packages',
        ]
        for base_path in common_paths:
            if os.path.exists(base_path):
                sdk_path = os.path.join(base_path, 'openhands', 'sdk')
                if os.path.exists(sdk_path):
                    print(sdk_path)
                    sys.exit(0)
        
        # Method 5: Search in sys.path
        for path in sys.path:
            if path and os.path.exists(path):
                sdk_path = os.path.join(path, 'openhands', 'sdk')
                if os.path.exists(sdk_path):
                    print(sdk_path)
                    sys.exit(0)
    except Exception:
        pass
    
    if attempt < 14:
        time.sleep(0.5)
PYEOF
)

if [ -n "$SDK_DIR" ] && [ -d "$SDK_DIR" ]; then
    echo "✅ Found SDK directory: $SDK_DIR"
    
    # Check if patches exist (try both possible locations)
    PATCHES_DIR=""
    if [ -d "/tmp/sdk_patches/openhands/sdk/openhands/sdk" ]; then
        PATCHES_DIR="/tmp/sdk_patches/openhands/sdk/openhands/sdk"
    elif [ -d "/tmp/sdk_patches/openhands/sdk" ]; then
        PATCHES_DIR="/tmp/sdk_patches/openhands/sdk"
    fi
    
    if [ -n "$PATCHES_DIR" ]; then
        echo "📦 Applying SDK patches from $PATCHES_DIR..."
        
        # Apply fn_call_converter.py patch
        if [ -f "$PATCHES_DIR/llm/mixins/fn_call_converter.py" ]; then
            # Try multiple methods to find the file
            TARGET_FILE=""
            
            # Method 1: Try importlib (works if module is already loaded)
            TARGET_FILE=$(python3 << 'PYEOF'
import importlib.util
import os

spec = importlib.util.find_spec('openhands.sdk.llm.mixins.fn_call_converter')
if spec and spec.origin and os.path.exists(spec.origin):
    print(spec.origin)
PYEOF
)
            
            # Method 2: Try site-packages directly (works before module is loaded)
            if [ -z "$TARGET_FILE" ] || [ ! -f "$TARGET_FILE" ]; then
                TARGET_FILE=$(python3 << 'PYEOF'
import site
import os

for sp in site.getsitepackages():
    file_path = os.path.join(sp, 'openhands', 'sdk', 'llm', 'mixins', 'fn_call_converter.py')
    if os.path.exists(file_path):
        print(file_path)
        break
PYEOF
)
            fi
            
            # Method 3: Fallback to SDK_DIR
            if [ -z "$TARGET_FILE" ] || [ ! -f "$TARGET_FILE" ]; then
                TARGET_FILE="$SDK_DIR/llm/mixins/fn_call_converter.py"
            fi
            
            if [ -n "$TARGET_FILE" ] && [ -f "$TARGET_FILE" ]; then
                echo "  → Patching fn_call_converter.py at: $TARGET_FILE"
                echo "  → Backup original file..."
                cp "$TARGET_FILE" "${TARGET_FILE}.backup" 2>/dev/null || true
                cp "$PATCHES_DIR/llm/mixins/fn_call_converter.py" "$TARGET_FILE"
                echo "  ✅ fn_call_converter.py patched"
                
                # Verify patch was applied
                if [ -f "$TARGET_FILE" ]; then
                    if grep -q 'message.get("content")' "$TARGET_FILE"; then
                        echo "  ✅ Verification: File contains .get() method"
                    fi
                    if grep -q 'or ""' "$TARGET_FILE" || grep -q 'if content is None:' "$TARGET_FILE"; then
                        echo "  ✅ Verification: File contains None check"
                    fi
                fi
                
                # Also patch in _MEIPASS if it exists (PyInstaller extracted files)
                # This is CRITICAL because PyInstaller loads modules from _MEIPASS
                # We need to patch _MEIPASS BEFORE the agent-server starts
                echo "  → Searching for _MEIPASS directories (PyInstaller)..."
                MEIPASS_FILES=$(python3 << 'PYEOF'
import os
import glob
import sys

found_files = []
found_dirs = []

# Method 1: Check if we're running in PyInstaller (sys._MEIPASS)
if getattr(sys, 'frozen', False):
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass and os.path.isdir(meipass):
        meipass_file = os.path.join(meipass, 'openhands', 'sdk', 'llm', 'mixins', 'fn_call_converter.py')
        if os.path.exists(meipass_file):
            found_files.append(meipass_file)
            found_dirs.append(meipass)
            print(f"Found sys._MEIPASS: {meipass_file}")

# Method 2: Check /tmp/_MEI* directories (PyInstaller temporary extraction)
# These are created when PyInstaller extracts files at runtime
meipass_dirs = glob.glob('/tmp/_MEI*')
for meipass_dir in meipass_dirs:
    if os.path.isdir(meipass_dir):
        meipass_file = os.path.join(meipass_dir, 'openhands', 'sdk', 'llm', 'mixins', 'fn_call_converter.py')
        if os.path.exists(meipass_file):
            if meipass_file not in found_files:
                found_files.append(meipass_file)
                found_dirs.append(meipass_dir)
                print(f"Found /tmp/_MEI*: {meipass_file}")
        else:
            # File doesn't exist, but directory does - we'll create it
            found_dirs.append(meipass_dir)
            print(f"Found /tmp/_MEI* directory (file not found): {meipass_dir}")

# Print all found files
for f in found_files:
    print(f"FILE:{f}")
# Print all found directories (for creating files if needed)
for d in found_dirs:
    print(f"DIR:{d}")
PYEOF
)
                # Process found files
                if [ -n "$MEIPASS_FILES" ]; then
                    MEIPASS_DIRS=$(echo "$MEIPASS_FILES" | grep "^DIR:" | sed 's/^DIR://')
                    MEIPASS_FILES_ONLY=$(echo "$MEIPASS_FILES" | grep "^FILE:" | sed 's/^FILE://')
                    
                    # Patch existing files
                    if [ -n "$MEIPASS_FILES_ONLY" ]; then
                        echo "  → Found $(echo "$MEIPASS_FILES_ONLY" | wc -l) _MEIPASS file(s)"
                        for MEIPASS_FILE in $MEIPASS_FILES_ONLY; do
                            if [ -f "$MEIPASS_FILE" ]; then
                                echo "  → Patching _MEIPASS: $MEIPASS_FILE"
                                # Backup original
                                cp "$MEIPASS_FILE" "${MEIPASS_FILE}.backup" 2>/dev/null || true
                                # Apply patch
                                cp "$PATCHES_DIR/llm/mixins/fn_call_converter.py" "$MEIPASS_FILE"
                                echo "  ✅ _MEIPASS file patched"
                                
                                # Verify patch was applied
                                if grep -q 'message.get("content")' "$MEIPASS_FILE"; then
                                    echo "  ✅ Verification: _MEIPASS file contains .get() method"
                                else
                                    echo "  ⚠️  Warning: _MEIPASS file may not be properly patched"
                                fi
                            fi
                        done
                    fi
                    
                    # Create files in directories where they don't exist
                    if [ -n "$MEIPASS_DIRS" ]; then
                        echo "  → Creating missing files in _MEIPASS directories..."
                        for MEIPASS_DIR in $MEIPASS_DIRS; do
                            MEIPASS_FILE="$MEIPASS_DIR/openhands/sdk/llm/mixins/fn_call_converter.py"
                            if [ ! -f "$MEIPASS_FILE" ]; then
                                echo "  → Creating file in _MEIPASS: $MEIPASS_FILE"
                                mkdir -p "$(dirname "$MEIPASS_FILE")"
                                cp "$PATCHES_DIR/llm/mixins/fn_call_converter.py" "$MEIPASS_FILE"
                                echo "  ✅ _MEIPASS file created and patched"
                                
                                # Verify patch was applied
                                if grep -q 'message.get("content")' "$MEIPASS_FILE"; then
                                    echo "  ✅ Verification: _MEIPASS file contains .get() method"
                                else
                                    echo "  ⚠️  Warning: _MEIPASS file may not be properly patched"
                                fi
                            fi
                        done
                    fi
                else
                    echo "  ℹ️  No _MEIPASS files or directories found yet (PyInstaller may create them later)"
                    echo "  → Will try to patch _MEIPASS again after agent-server starts"
                fi
                
                # Verify the patch was applied correctly
                echo "  → Verifying patch was applied..."
                if grep -q 'message.get("content")' "$TARGET_FILE" && ! grep -q 'message\["content"\]' "$TARGET_FILE"; then
                    echo "  ✅ Patch verification: file uses .get() method"
                else
                    echo "  ⚠️  Patch verification: file may still use direct access"
                fi
                
                # Check if module is already loaded and reload it
                echo "  → Checking if module is loaded..."
                LOADED_MODULE_PATH=$(python3 << 'PYEOF'
import sys
import importlib.util

module_name = 'openhands.sdk.llm.mixins.fn_call_converter'
if module_name in sys.modules:
    module = sys.modules[module_name]
    if hasattr(module, '__file__') and module.__file__:
        print(module.__file__)
    else:
        print("LOADED_BUT_NO_FILE")
else:
    print("NOT_LOADED")
PYEOF
)
                
                # Find and patch ALL instances of the module (including already loaded ones)
                echo "  → Searching for all module instances..."
                ALL_MODULE_FILES=$(python3 << 'PYEOF'
import sys
import importlib.util
import os
import glob

found_files = set()

# Method 1: Check if module is already loaded
module_name = 'openhands.sdk.llm.mixins.fn_call_converter'
if module_name in sys.modules:
    module = sys.modules[module_name]
    if hasattr(module, '__file__') and module.__file__:
        if os.path.exists(module.__file__):
            found_files.add(module.__file__)

# Method 2: Try to find via importlib
spec = importlib.util.find_spec(module_name)
if spec and spec.origin and os.path.exists(spec.origin):
    found_files.add(spec.origin)

# Method 3: Check _MEIPASS (PyInstaller)
if getattr(sys, 'frozen', False):
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        meipass_file = os.path.join(meipass, 'openhands', 'sdk', 'llm', 'mixins', 'fn_call_converter.py')
        if os.path.exists(meipass_file):
            found_files.add(meipass_file)

# Method 4: Check /tmp/_MEI* directories
for meipass_dir in glob.glob('/tmp/_MEI*'):
    if os.path.isdir(meipass_dir):
        meipass_file = os.path.join(meipass_dir, 'openhands', 'sdk', 'llm', 'mixins', 'fn_call_converter.py')
        if os.path.exists(meipass_file):
            found_files.add(meipass_file)

# Method 5: Check site-packages
import site
for sp in site.getsitepackages():
    file_path = os.path.join(sp, 'openhands', 'sdk', 'llm', 'mixins', 'fn_call_converter.py')
    if os.path.exists(file_path):
        found_files.add(file_path)

# Print all found files
for f in found_files:
    print(f)
PYEOF
)
                
                # Patch all found instances
                if [ -n "$ALL_MODULE_FILES" ]; then
                    INSTANCE_COUNT=$(echo "$ALL_MODULE_FILES" | wc -l)
                    echo "  → Found $INSTANCE_COUNT instance(s) of the module"
                    for MODULE_FILE in $ALL_MODULE_FILES; do
                        if [ "$MODULE_FILE" != "$TARGET_FILE" ] && [ -f "$MODULE_FILE" ]; then
                            echo "  → Patching additional instance: $MODULE_FILE"
                            cp "$PATCHES_DIR/llm/mixins/fn_call_converter.py" "$MODULE_FILE"
                            echo "  ✅ Additional instance patched"
                        fi
                    done
                fi
                
                # Reload the module to apply changes (if Python is running and module is loaded)
                echo "  → Reloading module..."
                python3 << 'PYEOF'
import sys
import importlib
import os

# Invalidate caches first
importlib.invalidate_caches()

# Try to reload the module if it's already loaded
module_name = 'openhands.sdk.llm.mixins.fn_call_converter'
if module_name in sys.modules:
    try:
        # Also reload dependent modules
        dependent_modules = [
            'openhands.sdk.llm.mixins.non_native_fc',
            'openhands.sdk.llm.llm',
        ]
        for dep_module in dependent_modules:
            if dep_module in sys.modules:
                try:
                    importlib.reload(sys.modules[dep_module])
                except Exception:
                    pass
        
        # Reload the main module
        importlib.reload(sys.modules[module_name])
        print(f"  ✅ Reloaded module: {module_name}")
        
        # Verify the reloaded module has the fix
        if hasattr(sys.modules[module_name], 'convert_fncall_messages_to_non_fncall_messages'):
            print(f"  ✅ Module function available after reload")
        
        # Verify the file was actually patched
        module_file = sys.modules[module_name].__file__
        if module_file and os.path.exists(module_file):
            with open(module_file, 'r') as f:
                content = f.read()
                if 'message.get("content")' in content or 'message.get(\'content\')' in content:
                    print(f"  ✅ Verified: Patched file contains .get() method")
                else:
                    print(f"  ⚠️  Warning: Patched file may not contain .get() method")
    except Exception as e:
        print(f"  ⚠️  Could not reload module: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  ℹ️  Module not yet loaded, will use patched file on first import")
PYEOF
            else
                echo "  ⚠️  Target file not found: $TARGET_FILE"
                echo "     Tried importlib, site-packages, and SDK_DIR methods"
            fi
        fi
        
        # Apply task_tracker.py patch
        if [ -f "$PATCHES_DIR/tools/task_tracker.py" ]; then
            # Try to find the actual file location via importlib
            TARGET_FILE=$(python3 << 'PYEOF'
import importlib.util
import os

spec = importlib.util.find_spec('openhands.sdk.tools.task_tracker')
if spec and spec.origin and os.path.exists(spec.origin):
    print(spec.origin)
PYEOF
)
            
            if [ -n "$TARGET_FILE" ] && [ -f "$TARGET_FILE" ]; then
                echo "  → Patching task_tracker.py at: $TARGET_FILE"
                cp "$PATCHES_DIR/tools/task_tracker.py" "$TARGET_FILE"
                echo "  ✅ task_tracker.py patched"
            else
                # Fallback: try the SDK_DIR location
                TARGET_FILE="$SDK_DIR/tools/task_tracker.py"
                if [ -f "$TARGET_FILE" ]; then
                    echo "  → Patching task_tracker.py (fallback)..."
                    cp "$PATCHES_DIR/tools/task_tracker.py" "$TARGET_FILE"
                    echo "  ✅ task_tracker.py patched"
                else
                    echo "  ⚠️  Target file not found: $TARGET_FILE"
                    echo "     Tried importlib method and SDK_DIR fallback"
                fi
            fi
        fi
        
        # Apply bash.py patch (if exists)
        if [ -f "$PATCHES_DIR/tools/bash.py" ]; then
            TARGET_FILE="$SDK_DIR/tools/bash.py"
            if [ -f "$TARGET_FILE" ]; then
                echo "  → Patching bash.py..."
                cp "$PATCHES_DIR/tools/bash.py" "$TARGET_FILE"
                echo "  ✅ bash.py patched"
            fi
        fi
        
        echo "✅ SDK patches applied successfully"
    else
        echo "⚠️  SDK patches directory not found"
        echo "   Checked: /tmp/sdk_patches/openhands/sdk/openhands/sdk"
        echo "   Checked: /tmp/sdk_patches/openhands/sdk"
    fi
else
    echo "⚠️  Could not find SDK directory"
    echo "   Trying to locate SDK manually..."
    python3 << 'PYEOF'
import sys
import os
import site

print("Python path:", sys.path)
print("Site packages:", site.getsitepackages())

# Try to find SDK
for path in sys.path:
    sdk_path = os.path.join(path, 'openhands', 'sdk')
    if os.path.exists(sdk_path):
        print(f"Found SDK at: {sdk_path}")
        break
PYEOF
fi


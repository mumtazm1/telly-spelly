#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
import sys
import subprocess
import grp

def check_pip():
    try:
        import pip
        return True
    except ImportError:
        return False

def check_input_group():
    """Check if user is in the input group"""
    try:
        input_group = grp.getgrnam('input')
        username = os.environ.get('USER', '')
        if username in input_group.gr_mem:
            return True
        # Also check if input is user's primary group
        if os.getgid() == input_group.gr_gid:
            return True
    except KeyError:
        pass
    return False

def install_requirements():
    """Install required Python packages"""
    if not check_pip():
        print("pip is not installed. Please install pip first.")
        return False

    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', '-r', 'requirements.txt'])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        return False

def install_application():
    # Define paths
    home = Path.home()
    app_name = "telly-spelly"
    
    # Check and install requirements first
    if not install_requirements():
        print("Failed to install required packages. Installation aborted.")
        return False
    
    # Create application directories
    app_dir = home / ".local/share/telly-spelly"
    bin_dir = home / ".local/bin"
    desktop_dir = home / ".local/share/applications"
    icon_dir = home / ".local/share/icons/hicolor/256x256/apps"
    
    # Create directories if they don't exist
    for directory in [app_dir, bin_dir, desktop_dir, icon_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Copy application files
    python_files = ["main.py", "recorder.py", "transcriber.py", "settings.py", 
                   "progress_window.py", "processing_window.py", "settings_window.py",
                   "loading_window.py", "shortcuts.py", "volume_meter.py"]
    
    for file in python_files:
        if os.path.exists(file):
            shutil.copy2(file, app_dir)
        else:
            print(f"Warning: Could not find {file}")
    
    # Copy requirements.txt
    if os.path.exists('requirements.txt'):
        shutil.copy2('requirements.txt', app_dir)
    
    # Create launcher script with proper Python path and input group handling
    launcher_path = bin_dir / app_name
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    # Detect if using pyenv
    python_path = shutil.which('python3')
    using_pyenv = '.pyenv' in (python_path or '')

    with open(launcher_path, 'w') as f:
        f.write(f'''#!/bin/bash

# Setup Python environment
export PYTHONPATH="$HOME/.local/lib/python{python_version}/site-packages:$PYTHONPATH"
''')
        if using_pyenv:
            f.write('''
# Setup pyenv if available
if [ -d "$HOME/.pyenv" ]; then
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
fi
''')
        f.write(f'''
cd {app_dir}

# Use sg to get input group access for keyboard shortcuts
exec sg input -c "python3 {app_dir}/main.py $*"
''')
    
    # Make launcher executable
    launcher_path.chmod(0o755)
    
    # Copy desktop file
    desktop_file = "org.kde.telly_spelly.desktop"
    if os.path.exists(desktop_file):
        shutil.copy2(desktop_file, desktop_dir)
    else:
        print(f"Warning: Could not find {desktop_file}")
    
    # Copy icon
    icon_file = "telly-spelly.png"
    if os.path.exists(icon_file):
        shutil.copy2(icon_file, icon_dir)
    else:
        print(f"Warning: Could not find {icon_file}")
    
    print("\nInstallation completed!")
    print(f"Application installed to: {app_dir}")

    # Check input group membership
    if not check_input_group():
        print("\n" + "="*60)
        print("IMPORTANT: Global keyboard shortcuts require input group access!")
        print("="*60)
        print("\nRun the following command to add yourself to the input group:")
        print(f"\n    sudo usermod -aG input $USER\n")
        print("Then LOG OUT and LOG BACK IN for the change to take effect.")
        print("="*60)
    else:
        print("\nInput group membership detected.")

    print("\nYou may need to log out and back in for the application to appear in your menu")
    return True

if __name__ == "__main__":
    try:
        success = install_application()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Installation failed: {e}")
        sys.exit(1) 

# Telly Spelly - Development Notes

## Project Overview

A KDE Plasma voice transcription tool using faster-whisper. Forked and customized from `gbasilveira/telly-spelly`.

## Important: Development vs Installed Version

⚠️ **This project has TWO copies of the code:**

| Location | Purpose |
|----------|---------|
| `/home/owais/Projects/telly-spelly/` | Development source code |
| `~/.local/share/telly-spelly/` | Installed/running version |

### After Making Changes

**ALWAYS reinstall after modifying source files:**

```bash
cd /home/owais/Projects/telly-spelly
python3 install.py
```

The desktop launcher runs from `~/.local/share/telly-spelly/`, NOT from this project directory. Changes to source files won't take effect until reinstalled.

### Quick Test Without Reinstalling

To test changes without reinstalling, run directly from the project:

```bash
cd /home/owais/Projects/telly-spelly
python3 main.py
```

---

## Custom Changes Made

### 1. Switched to faster-whisper (from openai-whisper)

**File:** `transcriber.py`

faster-whisper is significantly faster and uses less memory than the original openai-whisper library.

### 2. Switched to pynput (from keyboard library)

**File:** `shortcuts.py`

pynput works better on Linux without requiring root access in many cases.

### 3. Custom Dictionary Support

**Files modified:**
- `settings.py` - Added `get_config_dir()` and `get_custom_words_path()` helpers
- `transcriber.py` - Added `load_custom_words()`, `apply_replacements()`, integrated hotwords/initial_prompt

**Config file location:** `~/.config/telly-spelly/custom_words.json`

**Features:**
- **Hotwords**: Space-separated terms passed to faster-whisper to boost recognition
- **Replacements**: Post-processing dictionary to fix common misrecognitions (case-insensitive)
- **Initial prompt**: Context string to prime the model for specific domains

**Example config:**
```json
{
  "hotwords": "dbt Snowflake PostgreSQL Airflow Kafka",
  "replacements": {
    "dee bee tee": "dbt",
    "post gress": "PostgreSQL"
  },
  "initial_prompt": "Technical discussion about data engineering and ETL pipelines."
}
```

### 4. Smart Launcher with Input Group Handling

**File:** `install.py`

- Detects pyenv and sets up proper Python paths
- Uses `sg input` wrapper for input group access (global shortcuts)
- Logs startup to `~/.local/share/telly-spelly/launch.log`

### 5. Startup Input Group Check

**File:** `main.py`

Shows warning dialog if user lacks input group access (shortcuts won't work without it).

### 6. Added .gitignore

Ignores `__pycache__/`, `*.bak`, and other generated files.

---

## File Structure

```
telly-spelly/
├── main.py              # Application entry point
├── transcriber.py       # Whisper transcription (custom dictionary here)
├── recorder.py          # Audio recording
├── shortcuts.py         # Global keyboard shortcuts (pynput)
├── settings.py          # Settings management + config path helpers
├── settings_window.py   # Settings UI
├── loading_window.py    # Loading splash screen
├── progress_window.py   # Recording progress UI
├── processing_window.py # Processing status UI
├── clipboard_manager.py # Clipboard operations
├── volume_meter.py      # Volume visualization
├── window.py            # Base window utilities
├── install.py           # Installation script
├── uninstall.py         # Uninstallation script
├── requirements.txt     # Python dependencies
└── .gitignore           # Git ignore rules
```

---

## Git Remote Setup

This repo is configured to push to the fork, not the original:

```bash
# Verify remotes
git remote -v

# Should show:
# origin    https://github.com/mumtazm1/telly-spelly.git (fetch)
# origin    https://github.com/mumtazm1/telly-spelly.git (push)
```

---

## Debugging

### View launch log
```bash
cat ~/.local/share/telly-spelly/launch.log
```

### Run with full output
```bash
cd /home/owais/Projects/telly-spelly
python3 main.py 2>&1
```

### Check if custom words loaded
Look for these log lines on startup:
```
INFO:transcriber:Loaded 70 hotwords
INFO:transcriber:Loaded 31 replacements
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Loading forever" after code changes | Installed version is outdated | Run `python3 install.py` |
| Global shortcuts don't work | Not in input group | `sudo usermod -aG input $USER` then log out/in |
| JACK server warnings | Normal, not an error | Ignore these messages |

---

## Dependencies

```
PyQt6
numpy
pyaudio
scipy
pynput
faster-whisper
```

System packages needed:
```bash
# Ubuntu/Debian
sudo apt install python3-pyaudio portaudio19-dev

# Fedora
sudo dnf install python3-pyaudio portaudio-devel
```


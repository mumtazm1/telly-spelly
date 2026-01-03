from PyQt6.QtCore import QObject, pyqtSignal
from pynput import keyboard
import logging

logger = logging.getLogger(__name__)


class GlobalShortcuts(QObject):
    start_recording_triggered = pyqtSignal()
    stop_recording_triggered = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._listener = None
        self._start_keys = set()
        self._stop_keys = set()
        self._current_keys = set()

    def _parse_hotkey(self, hotkey_str):
        """Parse a hotkey string like 'ctrl+alt+r' into a set of pynput keys"""
        keys = set()
        for part in hotkey_str.lower().split('+'):
            part = part.strip()
            if part in ('ctrl', 'control'):
                keys.add(keyboard.Key.ctrl)
            elif part == 'alt':
                keys.add(keyboard.Key.alt)
            elif part == 'shift':
                keys.add(keyboard.Key.shift)
            elif part in ('super', 'meta', 'win', 'cmd'):
                keys.add(keyboard.Key.cmd)
            elif len(part) == 1:
                keys.add(keyboard.KeyCode.from_char(part))
            else:
                # Try to get special key
                try:
                    keys.add(getattr(keyboard.Key, part))
                except AttributeError:
                    keys.add(keyboard.KeyCode.from_char(part[0]))
        return keys

    def setup_shortcuts(self, start_key='ctrl+alt+r', stop_key='ctrl+alt+s'):
        """Setup global keyboard shortcuts using pynput"""
        try:
            self.remove_shortcuts()

            self._start_keys = self._parse_hotkey(start_key)
            self._stop_keys = self._parse_hotkey(stop_key)

            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            self._listener.start()

            logger.info(f"Global shortcuts registered - Start: {start_key}, Stop: {stop_key}")
            return True

        except Exception as e:
            logger.error(f"Failed to register global shortcuts: {e}")
            return False

    def remove_shortcuts(self):
        """Remove existing shortcuts"""
        if self._listener:
            self._listener.stop()
            self._listener = None
        self._current_keys.clear()

    def _normalize_key(self, key):
        """Normalize key for comparison (handle left/right variants)"""
        if hasattr(key, 'name'):
            name = key.name
            if name in ('ctrl_l', 'ctrl_r'):
                return keyboard.Key.ctrl
            elif name in ('alt_l', 'alt_r', 'alt_gr'):
                return keyboard.Key.alt
            elif name in ('shift_l', 'shift_r'):
                return keyboard.Key.shift
            elif name in ('cmd_l', 'cmd_r', 'super_l', 'super_r'):
                return keyboard.Key.cmd
        return key

    def _on_press(self, key):
        """Handle key press events"""
        normalized = self._normalize_key(key)
        self._current_keys.add(normalized)

        # Check if start hotkey is pressed
        if self._start_keys and self._start_keys.issubset(self._current_keys):
            logger.info("Start recording shortcut triggered")
            self.start_recording_triggered.emit()

        # Check if stop hotkey is pressed
        if self._stop_keys and self._stop_keys.issubset(self._current_keys):
            logger.info("Stop recording shortcut triggered")
            self.stop_recording_triggered.emit()

    def _on_release(self, key):
        """Handle key release events"""
        normalized = self._normalize_key(key)
        self._current_keys.discard(normalized)
        # Also discard the original key in case normalization differs
        self._current_keys.discard(key)

    def __del__(self):
        self.remove_shortcuts()

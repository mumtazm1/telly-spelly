import sys
import os
import signal
import atexit

# Check if CUDA libraries need to be set up and re-exec if necessary
def ensure_cuda_libs():
    """Ensure NVIDIA cuDNN libraries are in LD_LIBRARY_PATH, re-exec if needed"""
    try:
        import nvidia.cudnn
        import nvidia.cublas
        
        # Find library paths
        cudnn_lib = cublas_lib = None
        if hasattr(nvidia.cudnn, '__path__') and nvidia.cudnn.__path__:
            cudnn_lib = os.path.join(list(nvidia.cudnn.__path__)[0], 'lib')
        if hasattr(nvidia.cublas, '__path__') and nvidia.cublas.__path__:
            cublas_lib = os.path.join(list(nvidia.cublas.__path__)[0], 'lib')
        
        # Check if libraries are already in path
        current_ld = os.environ.get('LD_LIBRARY_PATH', '')
        needs_reexec = False
        new_paths = []
        
        if cudnn_lib and os.path.exists(cudnn_lib) and cudnn_lib not in current_ld:
            new_paths.append(cudnn_lib)
            needs_reexec = True
        if cublas_lib and os.path.exists(cublas_lib) and cublas_lib not in current_ld:
            new_paths.append(cublas_lib)
            needs_reexec = True
        
        # Re-exec with updated LD_LIBRARY_PATH if needed
        if needs_reexec and not os.environ.get('_CUDA_LIBS_SET'):
            new_ld = ':'.join(new_paths + ([current_ld] if current_ld else []))
            os.environ['LD_LIBRARY_PATH'] = new_ld
            os.environ['_CUDA_LIBS_SET'] = '1'
            os.execv(sys.executable, [sys.executable] + sys.argv)
            
    except ImportError:
        pass  # nvidia packages not installed

ensure_cuda_libs()

from PyQt6.QtWidgets import (QApplication, QMessageBox, QSystemTrayIcon, QMenu)
from PyQt6.QtCore import Qt, QTimer, QCoreApplication
from PyQt6.QtGui import QIcon, QAction
import logging
from settings_window import SettingsWindow
from progress_window import ProgressWindow
from processing_window import ProcessingWindow
from recorder import AudioRecorder
from transcriber import WhisperTranscriber
from loading_window import LoadingWindow
from PyQt6.QtCore import pyqtSignal
import warnings
import ctypes
from shortcuts import GlobalShortcuts
from settings import Settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress ALSA error messages
try:
    # Load ALSA error handler
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_char_p)

    def py_error_handler(filename, line, function, err, fmt):
        pass

    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

    # Set error handler
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except (OSError, AttributeError) as e:
    warnings.warn(f"Failed to suppress ALSA warnings: {e}", RuntimeWarning)

def check_input_group_access():
    """Check if we have access to input devices for global shortcuts"""
    try:
        # Check if we can access an input device
        import glob
        input_devices = glob.glob('/dev/input/event*')
        if input_devices and os.access(input_devices[0], os.R_OK):
            return True
    except Exception:
        pass
    return False

def kill_stale_telly_processes():
    """Kill any stale telly-spelly processes that might be holding GPU memory"""
    import subprocess
    current_pid = os.getpid()
    try:
        # Find other telly-spelly python processes
        result = subprocess.run(
            ['pgrep', '-f', 'python.*telly.*main.py'],
            capture_output=True, text=True
        )
        if result.stdout:
            for pid_str in result.stdout.strip().split('\n'):
                pid = int(pid_str)
                if pid != current_pid:
                    logger.warning(f"Killing stale telly-spelly process: {pid}")
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
    except Exception as e:
        logger.debug(f"Could not check for stale processes: {e}")

def check_dependencies():
    required_packages = ['faster_whisper', 'pyaudio', 'pynput']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
            logger.error(f"Failed to import required dependency: {package}")

    if missing_packages:
        error_msg = (
            "Missing required dependencies:\n"
            f"{', '.join(missing_packages)}\n\n"
            "Please install them using:\n"
            f"pip install {' '.join(missing_packages)}"
        )
        QMessageBox.critical(None, "Missing Dependencies", error_msg)
        return False

    # Check input group access
    if not check_input_group_access():
        logger.warning("No input device access - global shortcuts may not work")
        QMessageBox.warning(
            None,
            "Limited Functionality",
            "Global keyboard shortcuts may not work.\n\n"
            "To enable shortcuts, add yourself to the input group:\n"
            "    sudo usermod -aG input $USER\n\n"
            "Then log out and back in.\n\n"
            "The app will still work via the system tray icon."
        )

    return True

class TrayRecorder(QSystemTrayIcon):
    initialization_complete = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Initialize basic state
        self.recording = False
        self.settings_window = None
        self.progress_window = None
        self.processing_window = None
        self.recorder = None
        self.transcriber = None
        
        # Set tooltip
        self.setToolTip("Telly Spelly")
        
        # Enable activation by left click
        self.activated.connect(self.on_activate)
        
        # Add shortcuts handler
        # Use QueuedConnection for thread-safe signal delivery from pynput thread
        self.shortcuts = GlobalShortcuts()
        self.shortcuts.start_recording_triggered.connect(
            self.start_recording, Qt.ConnectionType.QueuedConnection)
        self.shortcuts.stop_recording_triggered.connect(
            self.stop_recording, Qt.ConnectionType.QueuedConnection)

    def initialize(self):
        """Initialize the tray recorder after showing loading window"""
        # Set application icon
        self.app_icon = QIcon.fromTheme("telly-spelly")
        if self.app_icon.isNull():
            # Fallback to theme icons if custom icon not found
            self.app_icon = QIcon.fromTheme("media-record")
            logger.warning("Could not load telly-spelly icon, using system theme icon")
        
        # Load white tray icon (falls back to app_icon if not found)
        self.tray_icon = QIcon.fromTheme("telly-spelly-tray")
        if self.tray_icon.isNull():
            self.tray_icon = self.app_icon
            
        # Set the icon for the app window
        QApplication.instance().setWindowIcon(self.app_icon)
        # Use white tray icon for system tray
        self.setIcon(self.tray_icon)
        
        # Use tray icon for normal state and theme icon for recording
        self.normal_icon = self.tray_icon
        self.recording_icon = QIcon.fromTheme("media-playback-stop")
        
        # Create menu
        self.setup_menu()
        
        # Setup global shortcuts
        if not self.shortcuts.setup_shortcuts():
            logger.warning("Failed to register global shortcuts")
            
    def setup_menu(self):
        menu = QMenu()
        
        # Add recording action
        self.record_action = QAction("Start Recording", menu)
        self.record_action.triggered.connect(self.toggle_recording)
        menu.addAction(self.record_action)
        
        # Add settings action
        self.settings_action = QAction("Settings", menu)
        self.settings_action.triggered.connect(self.toggle_settings)
        menu.addAction(self.settings_action)
        
        # Add separator before quit
        menu.addSeparator()
        
        # Add quit action
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self.quit_application)
        menu.addAction(quit_action)
        
        # Set the context menu
        self.setContextMenu(menu)

    @staticmethod
    def isSystemTrayAvailable():
        return QSystemTrayIcon.isSystemTrayAvailable()

    def toggle_recording(self):
        if self.recording:
            # Stop recording
            self.recording = False
            self.record_action.setText("Start Recording")
            self.setIcon(self.normal_icon)
            
            # Update progress window before stopping recording
            if self.progress_window:
                self.progress_window.set_processing_mode()
                self.progress_window.set_status("Processing audio...")
            
            # Stop the actual recording
            if self.recorder:
                try:
                    self.recorder.stop_recording()
                except Exception as e:
                    logger.error(f"Error stopping recording: {e}")
                    if self.progress_window:
                        self.progress_window.close()
                        self.progress_window = None
                    return
        else:
            # Start recording
            self.recording = True
            # Show progress window
            if not self.progress_window:
                self.progress_window = ProgressWindow("Voice Recording")
                self.progress_window.stop_clicked.connect(self.stop_recording)
            self.progress_window.show()
            
            # Start recording
            self.record_action.setText("Stop Recording")
            self.setIcon(self.recording_icon)
            self.recorder.start_recording()

    def stop_recording(self):
        """Handle stopping the recording and starting processing"""
        if not self.recording:
            return
        
        logger.info("TrayRecorder: Stopping recording")
        self.toggle_recording()  # This is now safe since toggle_recording handles everything

    def toggle_settings(self):
        if not self.settings_window:
            self.settings_window = SettingsWindow()
            self.settings_window.shortcuts_changed.connect(self.update_shortcuts)
        
        if self.settings_window.isVisible():
            self.settings_window.hide()
        else:
            self.settings_window.show()
            
    def update_shortcuts(self, start_key, stop_key):
        """Update global shortcuts"""
        if self.shortcuts.setup_shortcuts(start_key, stop_key):
            self.showMessage("Shortcuts Updated", 
                           f"Start: {start_key}\nStop: {stop_key}",
                           self.normal_icon)

    def on_activate(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:  # Left click
            self.toggle_recording()

    def quit_application(self):
        # Stop recording if active
        if self.recording:
            self.stop_recording()

        # Cleanup transcriber FIRST (releases ~6GB GPU memory)
        if self.transcriber:
            self.transcriber.cleanup()
            self.transcriber = None

        # Cleanup recorder
        if self.recorder:
            self.recorder.cleanup()
            self.recorder = None

        # Close all windows
        if self.settings_window and self.settings_window.isVisible():
            self.settings_window.close()

        if self.progress_window and self.progress_window.isVisible():
            self.progress_window.close()

        # Quit the application
        QApplication.quit()

    def update_volume_meter(self, value):
        if self.progress_window and self.recording:
            self.progress_window.update_volume(value)
    
    def handle_recording_finished(self, audio_file):
        """Called when recording is saved to file"""
        logger.info("TrayRecorder: Recording finished, starting transcription")
        
        # Ensure progress window is in processing mode
        if self.progress_window:
            self.progress_window.set_processing_mode()
            self.progress_window.set_status("Starting transcription...")
        
        if self.transcriber:
            self.transcriber.transcribe_file(audio_file)
        else:
            logger.error("Transcriber not initialized")
            if self.progress_window:
                self.progress_window.close()
                self.progress_window = None
            QMessageBox.critical(None, "Error", "Transcriber not initialized")
    
    def handle_recording_error(self, error):
        """Handle recording errors"""
        logger.error(f"TrayRecorder: Recording error: {error}")
        QMessageBox.critical(None, "Recording Error", error)
        self.stop_recording()
        if self.progress_window:
            self.progress_window.close()
            self.progress_window = None
    
    def update_processing_status(self, status):
        if self.progress_window:
            self.progress_window.set_status(status)
    
    def handle_transcription_finished(self, text):
        if text:
            # Copy text to clipboard
            QApplication.clipboard().setText(text)
            self.showMessage("Transcription Complete", 
                           "Text has been copied to clipboard",
                           self.normal_icon)
        
        # Close the progress window
        if self.progress_window:
            self.progress_window.close()
            self.progress_window = None
    
    def handle_transcription_error(self, error):
        QMessageBox.critical(None, "Transcription Error", error)
        if self.progress_window:
            self.progress_window.close()
            self.progress_window = None

    def start_recording(self):
        """Start a new recording"""
        if not self.recording:
            self.toggle_recording()
            
    def stop_recording(self):
        """Stop current recording"""
        if self.recording:
            self.toggle_recording()

def setup_application_metadata():
    QCoreApplication.setApplicationName("Telly Spelly")
    QCoreApplication.setApplicationVersion("1.0")
    QCoreApplication.setOrganizationName("KDE")
    QCoreApplication.setOrganizationDomain("kde.org")

# Global reference for signal handler cleanup
_tray_instance = None

def cleanup_on_exit():
    """Cleanup function called on exit to ensure GPU memory is freed"""
    global _tray_instance
    if _tray_instance and _tray_instance.transcriber:
        logger.info("Atexit: Cleaning up transcriber...")
        _tray_instance.transcriber.cleanup()

def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT to ensure proper cleanup"""
    global _tray_instance
    logger.info(f"Received signal {signum}, cleaning up...")
    if _tray_instance:
        _tray_instance.quit_application()
    sys.exit(0)

def main():
    global _tray_instance
    try:
        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        atexit.register(cleanup_on_exit)

        app = QApplication(sys.argv)
        setup_application_metadata()

        # Kill any stale processes that might be holding GPU memory
        kill_stale_telly_processes()

        # Show loading window first
        loading_window = LoadingWindow()
        loading_window.show()
        app.processEvents()  # Force update of UI
        loading_window.set_status("Checking system requirements...")
        app.processEvents()  # Force update of UI

        # Check if system tray is available
        if not TrayRecorder.isSystemTrayAvailable():
            QMessageBox.critical(None, "Error",
                "System tray is not available. Please ensure your desktop environment supports system tray icons.")
            return 1

        # Create tray icon but don't initialize yet
        tray = TrayRecorder()
        _tray_instance = tray  # Store for signal handler
        
        # Connect loading window to tray initialization
        tray.initialization_complete.connect(loading_window.close)
        
        # Check dependencies in background
        loading_window.set_status("Checking dependencies...")
        app.processEvents()  # Force update of UI
        if not check_dependencies():
            return 1
        
        # Ensure the application doesn't quit when last window is closed
        app.setQuitOnLastWindowClosed(False)
        
        # Initialize tray in background
        QTimer.singleShot(100, lambda: initialize_tray(tray, loading_window, app))
        
        return app.exec()
        
    except Exception as e:
        logger.exception("Failed to start application")
        QMessageBox.critical(None, "Error", 
            f"Failed to start application: {str(e)}")
        return 1

def initialize_tray(tray, loading_window, app):
    try:
        # Initialize basic tray setup
        loading_window.set_status("Initializing application...")
        app.processEvents()
        tray.initialize()
        
        # Initialize recorder
        loading_window.set_status("Initializing audio system...")
        app.processEvents()
        tray.recorder = AudioRecorder()
        
        # Initialize transcriber
        loading_window.set_status("Loading Whisper model...")
        app.processEvents()
        tray.transcriber = WhisperTranscriber()
        
        # Connect signals
        loading_window.set_status("Setting up signal handlers...")
        app.processEvents()
        tray.recorder.volume_updated.connect(tray.update_volume_meter)
        tray.recorder.recording_finished.connect(tray.handle_recording_finished)
        tray.recorder.recording_error.connect(tray.handle_recording_error)
        
        tray.transcriber.transcription_progress.connect(tray.update_processing_status)
        tray.transcriber.transcription_finished.connect(tray.handle_transcription_finished)
        tray.transcriber.transcription_error.connect(tray.handle_transcription_error)
        
        # Make tray visible
        loading_window.set_status("Starting application...")
        app.processEvents()
        tray.setVisible(True)
        
        # Signal completion
        tray.initialization_complete.emit()
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        QMessageBox.critical(None, "Error", f"Failed to initialize application: {str(e)}")
        loading_window.close()

if __name__ == "__main__":
    sys.exit(main()) 
import sys
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
import os
from shortcuts import GlobalShortcuts
from settings import Settings
# from mic_debug import MicDebugWindow

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
except:
    warnings.warn("Failed to suppress ALSA warnings", RuntimeWarning)

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
        
        # Create debug window but don't show it
        # self.debug_window = MicDebugWindow()
        
        # Set tooltip
        self.setToolTip("Telly Spelly")
        
        # Enable activation by left click
        self.activated.connect(self.on_activate)
        
        # Add shortcuts handler
        self.shortcuts = GlobalShortcuts()
        self.shortcuts.start_recording_triggered.connect(self.start_recording)
        self.shortcuts.stop_recording_triggered.connect(self.stop_recording)

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
        
        # Add debug window action
        # self.debug_action = QAction("Show Debug Window", menu)
        # self.debug_action.triggered.connect(self.toggle_debug_window)
        # menu.addAction(self.debug_action)
        
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
        # Cleanup recorder
        if self.recorder:
            self.recorder.cleanup()
            self.recorder = None
        
        # Close all windows
        if self.settings_window and self.settings_window.isVisible():
            self.settings_window.close()
            
        if self.progress_window and self.progress_window.isVisible():
            self.progress_window.close()
            
        # Stop recording if active
        if self.recording:
            self.stop_recording()
            
        # Quit the application
        QApplication.quit()

    def update_volume_meter(self, value):
        # Update debug window first
        if hasattr(self, 'debug_window'):
            self.debug_window.update_values(value)
            
        # Then update volume meter as before
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

    def toggle_debug_window(self):
        """Toggle debug window visibility"""
        if self.debug_window.isVisible():
            self.debug_window.hide()
            self.debug_action.setText("Show Debug Window")
        else:
            self.debug_window.show()
            self.debug_action.setText("Hide Debug Window")

def setup_application_metadata():
    QCoreApplication.setApplicationName("Telly Spelly")
    QCoreApplication.setApplicationVersion("1.0")
    QCoreApplication.setOrganizationName("KDE")
    QCoreApplication.setOrganizationDomain("kde.org")

def main():
    try:
        app = QApplication(sys.argv)
        setup_application_metadata()
        
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
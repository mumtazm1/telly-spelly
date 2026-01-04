from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer
from faster_whisper import WhisperModel
import os
import logging
import json
import re
from settings import Settings

logger = logging.getLogger(__name__)


def load_custom_words():
    """Load custom words configuration from JSON file"""
    custom_words_path = Settings.get_custom_words_path()
    
    default_config = {
        "hotwords": "",
        "replacements": {},
        "initial_prompt": ""
    }
    
    if not custom_words_path.exists():
        return default_config
    
    try:
        with open(custom_words_path, 'r') as f:
            config = json.load(f)
            # Merge with defaults to ensure all keys exist
            return {**default_config, **config}
    except Exception as e:
        logger.warning(f"Failed to load custom words: {e}")
        return default_config


def apply_replacements(text, replacements):
    """Apply case-insensitive replacements to text"""
    if not replacements:
        return text
    
    for pattern, replacement in replacements.items():
        # Case-insensitive replacement
        text = re.sub(re.escape(pattern), replacement, text, flags=re.IGNORECASE)
    
    return text


class TranscriptionWorker(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model, audio_file, language='en', custom_words=None):
        super().__init__()
        self.model = model
        self.audio_file = audio_file
        self.language = language
        self.custom_words = custom_words or {}

    def run(self):
        try:
            if not os.path.exists(self.audio_file):
                raise FileNotFoundError(f"Audio file not found: {self.audio_file}")

            self.progress.emit("Loading audio file...")
            self.progress.emit("Processing audio with Whisper...")

            # Build transcribe kwargs with custom words support
            transcribe_kwargs = {
                "language": None if self.language == 'auto' else self.language,
                "beam_size": 5,
                "vad_filter": True  # Filters out silence for faster processing
            }
            
            # Add hotwords if configured
            hotwords = self.custom_words.get('hotwords', '')
            if hotwords:
                transcribe_kwargs['hotwords'] = hotwords
                logger.debug(f"Using hotwords: {hotwords[:50]}...")
            
            # Add initial prompt if configured
            initial_prompt = self.custom_words.get('initial_prompt', '')
            if initial_prompt:
                transcribe_kwargs['initial_prompt'] = initial_prompt
                logger.debug(f"Using initial prompt: {initial_prompt[:50]}...")

            # faster-whisper returns (segments, info)
            segments, info = self.model.transcribe(self.audio_file, **transcribe_kwargs)

            # Collect all segments into text
            text = " ".join([segment.text.strip() for segment in segments])

            if not text:
                raise ValueError("No text was transcribed")
            
            # Apply post-processing replacements
            replacements = self.custom_words.get('replacements', {})
            if replacements:
                text = apply_replacements(text, replacements)
                logger.debug("Applied custom word replacements")

            self.progress.emit("Transcription completed!")
            logger.info(f"Transcribed text: {text[:100]}...")
            self.finished.emit(text)

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            self.error.emit(f"Transcription failed: {str(e)}")
            self.finished.emit("")
        finally:
            try:
                if os.path.exists(self.audio_file):
                    os.remove(self.audio_file)
            except Exception as e:
                logger.error(f"Failed to remove temporary file: {e}")


class WhisperTranscriber(QObject):
    transcription_progress = pyqtSignal(str)
    transcription_finished = pyqtSignal(str)
    transcription_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.model = None
        self.worker = None
        self.custom_words = {}
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._cleanup_worker)
        self._cleanup_timer.setSingleShot(True)
        self.load_custom_words()
        self.load_model()
    
    def load_custom_words(self):
        """Load custom words configuration"""
        self.custom_words = load_custom_words()
        if self.custom_words.get('hotwords'):
            logger.info(f"Loaded {len(self.custom_words['hotwords'].split())} hotwords")
        if self.custom_words.get('replacements'):
            logger.info(f"Loaded {len(self.custom_words['replacements'])} replacements")

    def load_model(self):
        try:
            settings = Settings()
            model_name = settings.get('model', 'base')
            device = settings.get('device', 'cuda')
            compute_type = settings.get('compute_type', 'float16')
            
            logger.info(f"Loading faster-whisper model: {model_name} on {device} ({compute_type})")

            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type
            )
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _cleanup_worker(self):
        if self.worker:
            if self.worker.isFinished():
                self.worker.deleteLater()
                self.worker = None

    def transcribe(self, audio_file):
        """Transcribe audio file using faster-whisper"""
        try:
            settings = Settings()
            language = settings.get('language', 'auto')

            self.transcription_progress.emit("Processing audio...")

            # Build transcribe kwargs with custom words support
            transcribe_kwargs = {
                "language": None if language == 'auto' else language,
                "beam_size": 5,
                "vad_filter": True
            }
            
            # Add hotwords if configured
            hotwords = self.custom_words.get('hotwords', '')
            if hotwords:
                transcribe_kwargs['hotwords'] = hotwords
            
            # Add initial prompt if configured
            initial_prompt = self.custom_words.get('initial_prompt', '')
            if initial_prompt:
                transcribe_kwargs['initial_prompt'] = initial_prompt

            segments, info = self.model.transcribe(audio_file, **transcribe_kwargs)

            text = " ".join([segment.text.strip() for segment in segments])

            if not text:
                raise ValueError("No text was transcribed")
            
            # Apply post-processing replacements
            replacements = self.custom_words.get('replacements', {})
            if replacements:
                text = apply_replacements(text, replacements)

            self.transcription_progress.emit("Transcription completed!")
            logger.info(f"Transcribed text: {text[:100]}...")
            self.transcription_finished.emit(text)

            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except Exception as e:
                logger.error(f"Failed to remove temporary file: {e}")

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self.transcription_error.emit(str(e))

    def transcribe_file(self, audio_file):
        if self.worker and self.worker.isRunning():
            logger.warning("Transcription already in progress")
            return

        self.transcription_progress.emit("Starting transcription...")

        settings = Settings()
        language = settings.get('language', 'auto')

        self.worker = TranscriptionWorker(self.model, audio_file, language, self.custom_words)
        self.worker.finished.connect(self.transcription_finished)
        self.worker.progress.connect(self.transcription_progress)
        self.worker.error.connect(self.transcription_error)
        self.worker.finished.connect(lambda: self._cleanup_timer.start(1000))
        self.worker.start()

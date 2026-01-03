from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer
from faster_whisper import WhisperModel
import os
import logging
import time
from settings import Settings

logger = logging.getLogger(__name__)


class TranscriptionWorker(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model, audio_file, language='en'):
        super().__init__()
        self.model = model
        self.audio_file = audio_file
        self.language = language

    def run(self):
        try:
            if not os.path.exists(self.audio_file):
                raise FileNotFoundError(f"Audio file not found: {self.audio_file}")

            self.progress.emit("Loading audio file...")
            self.progress.emit("Processing audio with Whisper...")

            # faster-whisper returns (segments, info)
            segments, info = self.model.transcribe(
                self.audio_file,
                language=None if self.language == 'auto' else self.language,
                beam_size=5,
                vad_filter=True  # Filters out silence for faster processing
            )

            # Collect all segments into text
            text = " ".join([segment.text.strip() for segment in segments])

            if not text:
                raise ValueError("No text was transcribed")

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
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._cleanup_worker)
        self._cleanup_timer.setSingleShot(True)
        self.load_model()

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

            segments, info = self.model.transcribe(
                audio_file,
                language=None if language == 'auto' else language,
                beam_size=5,
                vad_filter=True
            )

            text = " ".join([segment.text.strip() for segment in segments])

            if not text:
                raise ValueError("No text was transcribed")

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

        self.worker = TranscriptionWorker(self.model, audio_file, language)
        self.worker.finished.connect(self.transcription_finished)
        self.worker.progress.connect(self.transcription_progress)
        self.worker.error.connect(self.transcription_error)
        self.worker.finished.connect(lambda: self._cleanup_timer.start(1000))
        self.worker.start()

from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer, Qt
from faster_whisper import WhisperModel
import os
import logging
import json
import re
import time
import gc
from settings import Settings

logger = logging.getLogger(__name__)

# #region agent log
def _get_debug_log_path():
    """Get debug log path in config directory"""
    config_dir = Settings.get_config_dir()
    return config_dir / "debug.log"

def _debug_log(hypothesis_id, location, message, data=None):
    import json as _json
    try:
        debug_path = _get_debug_log_path()
        entry = {"hypothesisId": hypothesis_id, "location": location, "message": message, "data": data or {}, "timestamp": int(time.time() * 1000), "sessionId": "debug-session"}
        with open(debug_path, "a") as f:
            f.write(_json.dumps(entry) + "\n")
    except OSError as e:
        logger.debug(f"Could not write debug log: {e}")
# #endregion


def load_custom_words():
    """Load custom words configuration from JSON file"""
    # #region agent log
    _t0 = time.time()
    # #endregion
    custom_words_path = Settings.get_custom_words_path()
    
    default_config = {
        "hotwords": "",
        "replacements": {},
        "initial_prompt": ""
    }
    
    if not custom_words_path.exists():
        # #region agent log
        _debug_log("E", "load_custom_words", "no_config_file", {"exists": False, "load_time_ms": (time.time()-_t0)*1000})
        # #endregion
        return default_config
    
    try:
        with open(custom_words_path, 'r') as f:
            config = json.load(f)
            # Merge with defaults to ensure all keys exist
            result = {**default_config, **config}
            # #region agent log
            _debug_log("E", "load_custom_words", "loaded_config", {"hotwords_len": len(result.get("hotwords","")), "replacements_count": len(result.get("replacements",{})), "initial_prompt_len": len(result.get("initial_prompt","")), "load_time_ms": (time.time()-_t0)*1000})
            # #endregion
            return result
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
                "beam_size": 1,  # Changed from 5 to 1 for faster greedy decoding
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

            # #region agent log
            import wave as _wave
            try:
                with _wave.open(self.audio_file, 'rb') as wf:
                    _audio_duration = wf.getnframes() / wf.getframerate()
                    _audio_size = os.path.getsize(self.audio_file)
            except (OSError, _wave.Error) as e:
                logger.debug(f"Could not read audio file for debug logging: {e}")
                _audio_duration, _audio_size = -1, -1
            _debug_log("F,G,H,I", "TranscriptionWorker.run", "before_transcribe", {"has_hotwords": bool(hotwords), "hotwords_len": len(hotwords), "has_initial_prompt": bool(initial_prompt), "initial_prompt_len": len(initial_prompt), "audio_duration_sec": _audio_duration, "audio_size_bytes": _audio_size, "beam_size": transcribe_kwargs.get("beam_size")})
            _t_transcribe_start = time.time()
            # #endregion
            # faster-whisper returns (segments, info)
            segments, info = self.model.transcribe(self.audio_file, **transcribe_kwargs)

            # #region agent log
            _t_segments_start = time.time()
            # #endregion
            # Collect all segments into text
            text = " ".join([segment.text.strip() for segment in segments])
            # Explicitly delete segments to free GPU memory
            del segments
            del info
            # #region agent log
            _t_segments_end = time.time()
            _debug_log("A,B,C", "TranscriptionWorker.run", "after_transcribe_and_segments", {"transcribe_call_time_ms": (_t_segments_start - _t_transcribe_start)*1000, "segments_collect_time_ms": (_t_segments_end - _t_segments_start)*1000, "text_len": len(text)})
            # #endregion

            if not text:
                raise ValueError("No text was transcribed")
            
            # Apply post-processing replacements
            replacements = self.custom_words.get('replacements', {})
            # #region agent log
            _t_replace_start = time.time()
            # #endregion
            if replacements:
                text = apply_replacements(text, replacements)
                logger.debug("Applied custom word replacements")
            # #region agent log
            _debug_log("D", "TranscriptionWorker.run", "after_replacements", {"replacements_count": len(replacements), "replace_time_ms": (time.time() - _t_replace_start)*1000})
            # #endregion

            self.progress.emit("Transcription completed!")
            logger.info(f"Transcribed text: {text[:100]}...")
            self.finished.emit(text)

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            self.error.emit(f"Transcription failed: {str(e)}")
            self.finished.emit("")
        finally:
            # Clean up audio file
            try:
                if os.path.exists(self.audio_file):
                    os.remove(self.audio_file)
            except Exception as e:
                logger.error(f"Failed to remove temporary file: {e}")

            # Free GPU memory after transcription
            # Force garbage collection to release segment references
            gc.collect()

            # Note: ctranslate2 doesn't expose a cache clearing API
            # The memory is managed by ctranslate2's internal allocator
            # torch.cuda.empty_cache() won't help here since ctranslate2 uses
            # its own CUDA allocator, not PyTorch's


class WhisperTranscriber(QObject):
    transcription_progress = pyqtSignal(str)
    transcription_finished = pyqtSignal(str)
    transcription_error = pyqtSignal(str)

    # Reload model after this many transcriptions to prevent memory fragmentation
    MODEL_RELOAD_INTERVAL = 50

    def __init__(self):
        super().__init__()
        self.model = None
        self.worker = None
        self.custom_words = {}
        self._transcription_count = 0
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._cleanup_worker)
        self._cleanup_timer.setSingleShot(True)
        self.load_custom_words()
        self.load_model()

    def _schedule_worker_cleanup(self):
        """Schedule worker cleanup (called when worker finishes)"""
        self._cleanup_timer.start(1000)

    def cleanup(self):
        """Explicitly release the model and free GPU memory"""
        logger.info("Cleaning up transcriber and releasing GPU memory...")

        # Wait for any running worker to finish
        if self.worker and self.worker.isRunning():
            self.worker.wait(5000)  # Wait up to 5 seconds
            self.worker = None

        # Delete the model explicitly - this is the main GPU memory holder
        if self.model is not None:
            # Clear the model's internal references
            if hasattr(self.model, 'model'):
                del self.model.model
            del self.model
            self.model = None

        # Force garbage collection to release all Python references
        gc.collect()

        # Note: ctranslate2 doesn't expose a cache clearing API
        # Memory is released when the model object is deleted

        self._transcription_count = 0
        logger.info("Transcriber cleanup complete")

    def reload_model(self):
        """Reload the model to free any accumulated GPU memory"""
        # Safety check: don't reload while worker is using the model
        if self.worker and self.worker.isRunning():
            logger.warning("Cannot reload model while transcription is in progress")
            return

        logger.info(f"Reloading model after {self._transcription_count} transcriptions...")

        # Clean up the old model
        if self.model is not None:
            if hasattr(self.model, 'model'):
                del self.model.model
            del self.model
            self.model = None

        gc.collect()

        # Load fresh model
        self.load_model()
        self._transcription_count = 0
        logger.info("Model reloaded successfully")

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
            
            # #region agent log
            _debug_log("F,G,H", "load_model", "model_settings", {"model_name": model_name, "device": device, "compute_type": compute_type})
            # #endregion
            
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
                "beam_size": 1,  # Changed from 5 to 1 for faster greedy decoding
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
            # Explicitly delete to free GPU memory
            del segments
            del info

            if not text:
                raise ValueError("No text was transcribed")
            
            # Apply post-processing replacements
            replacements = self.custom_words.get('replacements', {})
            if replacements:
                text = apply_replacements(text, replacements)

            self.transcription_progress.emit("Transcription completed!")
            logger.info(f"Transcribed text: {text[:100]}...")
            self.transcription_finished.emit(text)

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self.transcription_error.emit(str(e))
        finally:
            # Clean up audio file
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except OSError:
                pass
            # Free GPU memory
            gc.collect()
            # Note: torch.cuda.empty_cache() is ineffective for ctranslate2
            # since ctranslate2 uses its own CUDA allocator

    def transcribe_file(self, audio_file):
        if self.worker and self.worker.isRunning():
            logger.warning("Transcription already in progress")
            return

        # Check if model needs reloading to prevent memory accumulation
        self._transcription_count += 1
        if self._transcription_count >= self.MODEL_RELOAD_INTERVAL:
            logger.info(f"Transcription count ({self._transcription_count}) reached reload interval")
            self.reload_model()

        # #region agent log
        _debug_log("E", "transcribe_file", "start", {"audio_file": audio_file, "custom_words_cached": bool(self.custom_words), "hotwords_in_cache": len(self.custom_words.get("hotwords","")), "transcription_count": self._transcription_count})
        # #endregion
        self.transcription_progress.emit("Starting transcription...")

        settings = Settings()
        language = settings.get('language', 'auto')

        self.worker = TranscriptionWorker(self.model, audio_file, language, self.custom_words)
        # Use QueuedConnection for thread-safe signal delivery from worker thread
        self.worker.finished.connect(self.transcription_finished, Qt.ConnectionType.QueuedConnection)
        self.worker.progress.connect(self.transcription_progress, Qt.ConnectionType.QueuedConnection)
        self.worker.error.connect(self.transcription_error, Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self._schedule_worker_cleanup, Qt.ConnectionType.QueuedConnection)
        self.worker.start()

import os
import glob
import atexit
import time
import random
from google.genai import types
from google.genai import Client
import json
import tempfile
import logging
import datetime
import re
import statistics
import requests
from flask import Flask, request, render_template, jsonify, send_file, url_for
from google.cloud import speech
from google.cloud import storage
from google.cloud import texttospeech
from google.api_core import exceptions
from fuzzywuzzy import fuzz
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure Cloud Storage - Get bucket name from environment variable
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'strawberry-cupcake-files')
storage_client = storage.Client()

# Configure tracking webhook URL
TRACKING_WEBHOOK_URL = os.environ.get('TRACKING_WEBHOOK_URL', '')

def get_or_create_bucket(bucket_name):
    """Obtiene un bucket existente o crea uno nuevo."""
    try:
        # Inicializa el cliente con autenticación implícita
        storage_client = storage.Client()
        
        # Intenta obtener el bucket
        try:
            bucket = storage_client.get_bucket(bucket_name)
            logger.info(f"Conexión exitosa con el bucket: {bucket_name}")
            return bucket
        except exceptions.NotFound:
            # El bucket no existe, intenta crearlo
            logger.info(f"Bucket {bucket_name} no encontrado, intentando crearlo...")
            try:
                bucket = storage_client.create_bucket(bucket_name, location="us-central1")
                logger.info(f"Bucket {bucket_name} creado exitosamente.")
                return bucket
            except Exception as e:
                logger.error(f"Error al crear el bucket {bucket_name}: {str(e)}")
                return None
    except Exception as e:
        logger.error(f"Error al inicializar conexión con Storage: {str(e)}")
        return None

bucket = get_or_create_bucket(BUCKET_NAME)

# Create uploads folder for local testing
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create TTS temp directory for audio files
TTS_TEMP_DIR = os.path.join(tempfile.gettempdir(), 'strawberry_tts')
os.makedirs(TTS_TEMP_DIR, exist_ok=True)

# TTS file max age in seconds (2 hours)
TTS_FILE_MAX_AGE = 2 * 60 * 60

def cleanup_old_tts_files():
    """Remove TTS files older than TTS_FILE_MAX_AGE"""
    try:
        current_time = time.time()
        for filepath in glob.glob(os.path.join(TTS_TEMP_DIR, 'tts_*.mp3')):
            try:
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > TTS_FILE_MAX_AGE:
                    os.remove(filepath)
                    logger.info(f"Cleaned up old TTS file: {filepath}")
            except OSError as e:
                logger.warning(f"Error removing TTS file {filepath}: {e}")
    except Exception as e:
        logger.warning(f"Error during TTS cleanup: {e}")

def cleanup_all_tts_files():
    """Remove all TTS files on shutdown"""
    try:
        for filepath in glob.glob(os.path.join(TTS_TEMP_DIR, 'tts_*.mp3')):
            try:
                os.remove(filepath)
            except OSError:
                pass
        logger.info("Cleaned up all TTS files on shutdown")
    except Exception as e:
        logger.warning(f"Error during TTS shutdown cleanup: {e}")

# Register cleanup on app shutdown
atexit.register(cleanup_all_tts_files)

# Cleanup old files on startup
cleanup_old_tts_files()

# Maximum file size (20MB)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'opus', 'webm', 'ogg'}

# Add X-Robots-Tag header to prevent search engine indexing
@app.after_request
def add_security_headers(response):
    """Add security and robots headers to all responses"""
    response.headers['X-Robots-Tag'] = 'noindex, nofollow, noarchive'
    return response

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load Spanish Dictionary for pronunciation assessment
def load_dictionary():
    """Load Spanish dictionary from either cloud storage or local file"""
    try:
        # First try to load from local file
        try:
            with open("es_50k.txt", "r", encoding="utf-8") as f:
                words = [line.strip().split()[0].lower() for line in f if line.strip()]
                return set(words)
        except FileNotFoundError:
            # If local file not found, try to load from Cloud Storage
            if bucket:
                blob = bucket.blob('es_50k.txt')
                try:
                    if blob.exists():
                        content = blob.download_as_string().decode('utf-8')
                        words = [line.strip().split()[0].lower() for line in content.splitlines() if line.strip()]
                        return set(words)
                except exceptions.NotFound:
                    logger.warning(f"Dictionary file not found in bucket {BUCKET_NAME}")
            # Fallback to a small built-in dictionary
            logger.warning("Could not load dictionary file. Using minimal built-in dictionary.")
            return set([
                "hola", "como", "estás", "bien", "gracias", "adios", "buenos", "días", 
                "hasta", "luego", "mañana", "tarde", "noche", "por", "favor", "de", "nada",
                "sí", "no", "tal", "vez", "quizás", "casa", "coche", "trabajo", "escuela",
                "universidad", "restaurante", "tienda", "mercado", "parque", "playa", "montaña",
                "emergencia", "calma", "siga", "instrucciones", "seguridad", "caso"
            ])
    except Exception as e:
        logger.error(f"Error loading dictionary: {e}")
        return set()

# Load reference phrases for assessment and practice
def load_references():
    """Load reference phrases from file or provide defaults"""
    try:
        try:
            with open("references.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            if bucket:
                blob = bucket.blob('references.json')
                try:
                    if blob.exists():
                        content = blob.download_as_string().decode('utf-8')
                        return json.loads(content)
                except exceptions.NotFound:
                    logger.warning(f"References file not found in bucket {BUCKET_NAME}")
            # Default references if file not found
            return {
                "short": "Hola, ¿cómo estás? Espero que estés teniendo un buen día.",
                "medium": "Los bomberos llegaron rápidamente al lugar del incendio.",
                "extended": "En caso de emergencia, mantenga la calma y siga las instrucciones de seguridad."
            }
    except Exception as e:
        logger.error(f"Error loading references: {e}")
        return {
            "short": "Hola, ¿cómo estás?",
            "medium": "Me gusta viajar y conocer nuevas culturas.",
            "extended": "La educación es fundamental para el desarrollo de la sociedad."
        }

# Load pronunciation profile criteria from configuration file
def load_profile_criteria():
    """Load pronunciation profile criteria from configuration file"""
    try:
        try:
            with open("actfl_criteria.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            if bucket:
                blob = bucket.blob('actfl_criteria.json')
                try:
                    if blob.exists():
                        content = blob.download_as_string().decode('utf-8')
                        return json.loads(content)
                except exceptions.NotFound:
                    logger.warning(f"Profile criteria file not found in bucket {BUCKET_NAME}")
            # Return None if file not found - will use built-in criteria
            logger.warning("Profile criteria file not found. Using built-in criteria.")
            return None
    except Exception as e:
        logger.error(f"Error loading profile criteria: {e}")
        return None

# Initialize Spanish dictionary, references, and pronunciation profile criteria
SPANISH_DICT = load_dictionary()
REFERENCES = load_references()
PROFILE_CRITERIA = load_profile_criteria()
logger.info(f"Dictionary loaded with {len(SPANISH_DICT)} words")
logger.info(f"Profile criteria loaded: {'Yes' if PROFILE_CRITERIA else 'No (using built-in)'}")

def transcribe_audio(audio_content):
    """Transcribe Spanish audio using Google Cloud Speech-to-Text with support for up to 2 minutes

    Returns:
        dict: {
            'transcript': str - Full transcribed text,
            'words': list - Word objects with timing and confidence data
        }
    """
    client = speech.SpeechClient()

    # Check audio size to determine which method to use
    # Conservative threshold: 200 KB ensures ~50-60 seconds at 32 kbps stays within
    # Google's synchronous recognize() limits, while longer recordings use long_running_recognize()
    SIZE_THRESHOLD = 200 * 1024  # 200 KB (conservative for reliability)
    audio_size = len(audio_content)

    logger.info(f"Audio size: {audio_size} bytes ({audio_size / 1024:.1f} KB)")

    # Calculate estimated duration based on 32 kbps bitrate
    estimated_duration = (audio_size * 8) / 32000  # seconds
    logger.info(f"Estimated duration at 32 kbps: {estimated_duration:.1f} seconds")

    # Configuration for audio recognition
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
        sample_rate_hertz=48000,
        language_code="es-ES",
        alternative_language_codes=["es-MX", "es-US"],
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,  # NEW: For phonetic analysis
        enable_word_confidence=True,     # NEW: For accuracy scoring
        use_enhanced=True,
        model="default",
        audio_channel_count=1
    )

    try:
        # For shorter audio (<=50 seconds at 32kbps), use fast inline recognize()
        if audio_size <= SIZE_THRESHOLD:
            logger.info(f"Using fast inline recognize() method (audio size: {audio_size / 1024:.1f} KB <= {SIZE_THRESHOLD / 1024:.0f} KB threshold)")
            audio = speech.RecognitionAudio(content=audio_content)
            response = client.recognize(config=config, audio=audio)

            if response.results:
                # Extract transcript and word data
                transcript_parts = []
                all_words = []

                for result in response.results:
                    alternative = result.alternatives[0]
                    transcript_parts.append(alternative.transcript)

                    # Extract word-level data with timing and confidence
                    if hasattr(alternative, 'words') and alternative.words:
                        for word_info in alternative.words:
                            word_data = {
                                'word': word_info.word,
                                'start_time': word_info.start_time.total_seconds() if hasattr(word_info.start_time, 'total_seconds') else 0,
                                'end_time': word_info.end_time.total_seconds() if hasattr(word_info.end_time, 'total_seconds') else 0,
                                'confidence': word_info.confidence if hasattr(word_info, 'confidence') else 0.9
                            }
                            all_words.append(word_data)

                transcript = " ".join(transcript_parts)
                logger.info(f"Inline transcription successful ({len(transcript)} chars): '{transcript[:100]}...'")
                logger.info(f"Extracted {len(all_words)} words with timing data")

                return {
                    'transcript': transcript,
                    'words': all_words
                }
            else:
                logger.warning("No transcription results from inline recognize()")
                return {
                    'transcript': '',
                    'words': []
                }

        # For longer audio (>50 seconds at 32kbps), use long_running_recognize() with Cloud Storage
        else:
            logger.info(f"Using long_running_recognize() method (audio size: {audio_size / 1024:.1f} KB > {SIZE_THRESHOLD / 1024:.0f} KB threshold, est. {estimated_duration:.1f}s)")

            if not bucket:
                logger.warning("Bucket not available for long audio transcription, attempting fallback to inline recognize()")
                # Fallback: try inline recognize() even though it might fail for very long audio
                try:
                    audio = speech.RecognitionAudio(content=audio_content)
                    response = client.recognize(config=config, audio=audio)
                    if response.results:
                        transcript_parts = []
                        all_words = []

                        for result in response.results:
                            alternative = result.alternatives[0]
                            transcript_parts.append(alternative.transcript)

                            if hasattr(alternative, 'words') and alternative.words:
                                for word_info in alternative.words:
                                    word_data = {
                                        'word': word_info.word,
                                        'start_time': word_info.start_time.total_seconds() if hasattr(word_info.start_time, 'total_seconds') else 0,
                                        'end_time': word_info.end_time.total_seconds() if hasattr(word_info.end_time, 'total_seconds') else 0,
                                        'confidence': word_info.confidence if hasattr(word_info, 'confidence') else 0.9
                                    }
                                    all_words.append(word_data)

                        transcript = " ".join(transcript_parts)
                        logger.info(f"Fallback inline transcription successful: '{transcript}'")
                        return {
                            'transcript': transcript,
                            'words': all_words
                        }
                    else:
                        logger.error("Fallback inline transcription returned no results")
                        return {
                            'transcript': '',
                            'words': []
                        }
                except Exception as fallback_error:
                    logger.error(f"Fallback inline transcription failed: {str(fallback_error)}")
                    return {
                        'transcript': '',
                        'words': []
                    }

            # Upload audio to Cloud Storage
            blob_name = f"temp_audio/{uuid.uuid4()}.webm"
            blob = bucket.blob(blob_name)
            blob.upload_from_bytes(audio_content)
            logger.info(f"Uploaded audio to gs://{BUCKET_NAME}/{blob_name}")

            # Create GCS URI
            gcs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
            audio = speech.RecognitionAudio(uri=gcs_uri)

            # Use long_running_recognize for audio up to 2 minutes
            operation = client.long_running_recognize(config=config, audio=audio)

            # Wait for operation to complete (timeout 300 seconds)
            response = operation.result(timeout=300)

            # Clean up temporary file
            try:
                blob.delete()
                logger.info(f"Deleted temporary file: {blob_name}")
            except Exception as cleanup_error:
                logger.warning(f"Could not delete temporary file: {cleanup_error}")

            if response.results:
                transcript_parts = []
                all_words = []

                for result in response.results:
                    alternative = result.alternatives[0]
                    transcript_parts.append(alternative.transcript)

                    if hasattr(alternative, 'words') and alternative.words:
                        for word_info in alternative.words:
                            word_data = {
                                'word': word_info.word,
                                'start_time': word_info.start_time.total_seconds() if hasattr(word_info.start_time, 'total_seconds') else 0,
                                'end_time': word_info.end_time.total_seconds() if hasattr(word_info.end_time, 'total_seconds') else 0,
                                'confidence': word_info.confidence if hasattr(word_info, 'confidence') else 0.9
                            }
                            all_words.append(word_data)

                transcript = " ".join(transcript_parts)
                logger.info(f"Long-running transcription successful ({len(transcript)} chars): '{transcript[:100]}...'")
                logger.info(f"Extracted {len(all_words)} words with timing data")

                return {
                    'transcript': transcript,
                    'words': all_words
                }
            else:
                logger.warning("No transcription results from long_running_recognize()")
                return {
                    'transcript': '',
                    'words': []
                }

    except Exception as e:
        # Check if this is a timeout-related exception
        is_timeout = (
            isinstance(e, (TimeoutError, exceptions.DeadlineExceeded)) or
            'timeout' in str(e).lower() or
            'deadline exceeded' in str(e).lower()
        )

        if is_timeout:
            logger.error(f"Timeout error during transcription: {str(e)}")
            return {
                'transcript': '',
                'words': []
            }

        logger.error(f"Error during transcription: {str(e)}")

        # For longer audio that failed, try fallback to standard recognize if size permits
        if audio_size > SIZE_THRESHOLD and audio_size <= 10 * 1024 * 1024:
            logger.info("Attempting fallback to standard recognize()")
            try:
                audio_inline = speech.RecognitionAudio(content=audio_content)
                response = client.recognize(config=config, audio=audio_inline)
                if response.results:
                    transcript_parts = []
                    all_words = []

                    for result in response.results:
                        alternative = result.alternatives[0]
                        transcript_parts.append(alternative.transcript)

                        if hasattr(alternative, 'words') and alternative.words:
                            for word_info in alternative.words:
                                word_data = {
                                    'word': word_info.word,
                                    'start_time': word_info.start_time.total_seconds() if hasattr(word_info.start_time, 'total_seconds') else 0,
                                    'end_time': word_info.end_time.total_seconds() if hasattr(word_info.end_time, 'total_seconds') else 0,
                                    'confidence': word_info.confidence if hasattr(word_info, 'confidence') else 0.9
                                }
                                all_words.append(word_data)

                    transcript = " ".join(transcript_parts)
                    logger.info(f"Fallback transcription successful: '{transcript}'")
                    return {
                        'transcript': transcript,
                        'words': all_words
                    }
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")

        return {
            'transcript': '',
            'words': []
        }

# =============================================================================
# FACT ASSESSMENT SYSTEM - Based on Instructor's Rubric
# =============================================================================

# INTERNAL ACTFL BAND MAPPING (invisible to users)
# Used for: adjusting expectations, weighting penalties, modulating feedback tone
ACTFL_BANDS = {
    'novice_low': {'min': 0, 'max': 49, 'name': 'Novice Low'},
    'novice_mid': {'min': 50, 'max': 54, 'name': 'Novice Mid'},
    'novice_high': {'min': 55, 'max': 59, 'name': 'Novice High'},
    'intermediate_low': {'min': 60, 'max': 64, 'name': 'Intermediate Low'},
    'intermediate_mid': {'min': 65, 'max': 69, 'name': 'Intermediate Mid'},
    'intermediate_high': {'min': 70, 'max': 74, 'name': 'Intermediate High'},
    'advanced_low': {'min': 75, 'max': 79, 'name': 'Advanced Low'},
    'advanced_mid': {'min': 80, 'max': 84, 'name': 'Advanced Mid'},
    'advanced_high': {'min': 85, 'max': 89, 'name': 'Advanced High'},
    'superior': {'min': 90, 'max': 94, 'name': 'Superior'},
    'distinguished': {'min': 95, 'max': 100, 'name': 'Distinguished'}
}

def get_actfl_band(score):
    """Map score to internal ACTFL band (never shown to user)"""
    for band_key, band_info in ACTFL_BANDS.items():
        if band_info['min'] <= score <= band_info['max']:
            return band_key
    return 'novice_low'  # fallback

# LEVEL CONFIGURATION - Expected signals by level (NOT requirements, just signals)
LEVEL_CONFIGS = {
    'beginner': {
        'name': 'Beginner',
        'expected_structures': {
            'present': r'\b(soy|estoy|tengo|hablo|vivo|trabajo|estudio|como|hago|voy|quiero|puedo|debo|me gusta|se llama)\b',
            'basic_modality': r'\b(me gusta|quiero|puedo|necesito)\b'
        },
        'expected_connectors': ['y', 'pero', 'también', 'porque'],
        'vocabulary': {
            'personal': ['yo', 'mi', 'me', 'mis', 'nombre', 'soy', 'tengo', 'familia', 'años', 'casa'],
            'thematic_progression': 'personal'  # Personal vocabulary = high score at this level
        },
        'prompt_checklist': {
            'introduce_yourself': ['name_origin', 'age_occupation', 'languages', 'hobbies']
        },
        'level_multiplier': 1.0  # Base multiplier for penalties
    },
    'intermediate': {
        'name': 'Intermediate',
        'expected_structures': {
            'preterite': r'\b(fui|hice|comí|dije|fue|hablé|estudié|trabajé|viví|tuve|estuvo|hizo|desperté)\b',
            'imperfect': r'\b(era|estaba|tenía|iba|hacía|hablaba|comía|vivía|trabajaba|estudiaba)\b',
            'temporal_markers': r'\b(ayer|primero|después|luego|entonces|cuando|mientras)\b',
            'cognitive_modality': r'\b(creo que|pienso que|me parece que|considero que)\b'
        },
        'expected_connectors': ['primero', 'después', 'luego', 'porque', 'pero', 'cuando', 'mientras', 'por la mañana', 'por la tarde'],
        'vocabulary': {
            'everyday': ['casa', 'trabajo', 'escuela', 'comer', 'estudiar', 'amigos', 'tiempo', 'día', 'hacer', 'ir'],
            'thematic_progression': 'everyday'  # Everyday vocabulary = high score at this level
        },
        'prompt_checklist': {
            'describe_your_day': ['wake_time_morning', 'activities', 'met_people', 'how_felt']
        },
        'level_multiplier': 1.5  # Moderate penalties for same issues
    },
    'advanced': {
        'name': 'Advanced',
        'expected_structures': {
            'subjunctive': r'\b(sea|esté|tenga|quiera|pueda|haya|espero que|es importante que|me preocupa que|no creo que|ojalá|para que)\b',
            'conditional': r'\b(sería|haría|iría|tendría|podría|debería|si fuera|si tuviera|si pudiera)\b',
            'evaluative_modality': r'\b(me parece importante|es necesario que|considero que|me preocupa que|no creo que)\b',
            'complex_connectors': r'\b(sin embargo|no obstante|por un lado|por otro lado|aunque|a pesar de|por lo tanto|debido a)\b'
        },
        'expected_connectors': ['sin embargo', 'no obstante', 'aunque', 'por un lado', 'por otro lado', 'por lo tanto', 'debido a', 'ya que'],
        'vocabulary': {
            'abstract': ['sociedad', 'cultura', 'problema', 'educación', 'importante', 'necesario', 'tecnología', 'futuro', 'desarrollar', 'híbrido'],
            'thematic_progression': 'abstract'  # Abstract vocabulary = high score at this level
        },
        'prompt_checklist': {
            'opinion_technology_education': ['positive_aspect', 'concern_doubt', 'personal_experience', 'future_idea']
        },
        'level_multiplier': 2.0  # Strong penalties for same issues at advanced level
    }
}

# MODALITY DETECTION - 3 layers
MODALITY_LAYERS = {
    'basic': {  # Basic preferences and abilities
        'patterns': r'\b(me gusta|me encanta|quiero|puedo|necesito|prefiero)\b',
        'weight': 0.3
    },
    'cognitive': {  # Cognitive/opinion modality
        'patterns': r'\b(creo que|pienso que|me parece que|considero que|opino que)\b',
        'weight': 0.6
    },
    'evaluative': {  # Evaluative/normative modality
        'patterns': r'\b(es importante que|es necesario que|me preocupa que|me alegra que|espero que|ojalá)\b',
        'weight': 1.0
    }
}

# CONNECTOR TYPES - For discourse organization
CONNECTOR_TYPES = {
    'temporal': ['primero', 'después', 'luego', 'entonces', 'cuando', 'mientras', 'antes', 'finalmente'],
    'causal': ['porque', 'por eso', 'ya que', 'debido a', 'por lo tanto'],
    'adversative': ['pero', 'aunque', 'sin embargo', 'no obstante', 'a pesar de'],
    'additive': ['y', 'también', 'además', 'asimismo'],
    'conclusive': ['finalmente', 'en resumen', 'en conclusión']
}

# =============================================================================
# CAPA 3 — DIAGNOSTIC LAYER (Pattern Activation & Prioritization)
# =============================================================================
# Key principle: Patterns NEVER affect score. They only interpret why clarity was limited.

# Complete pattern catalog (10 patterns total)
DIAGNOSTIC_PATTERNS = {
    # GROUP 1 — FUNDAMENTALS (Highest impact on intelligibility)
    1: {
        'id': 1,
        'name': 'Pure Vowels',
        'group': 'fundamentals',
        'group_priority': 1,
        'activated_by': ['vowel_dragging', 'wps_collapse', 'micro_pauses_in_syllables'],
        'diagnostic_message': 'Vowel length is reducing rhythmic stability',
        'coaching_target': 'pure_vowels'
    },
    2: {
        'id': 2,
        'name': 'Even Syllable Rhythm',
        'group': 'fundamentals',
        'group_priority': 1,
        'activated_by': ['unstable_rhythm', 'high_wps_std_dev'],
        'diagnostic_message': 'Rhythm varies too much between syllables',
        'coaching_target': 'even_syllable_rhythm'
    },

    # GROUP 2 — PROSODY (Discourse-level clarity)
    9: {
        'id': 9,
        'name': 'Intonation',
        'group': 'prosody',
        'group_priority': 2,
        'activated_by': ['flat_melodic_contour', 'no_boundary_marking'],
        'diagnostic_message': 'Ideas are not clearly marked through intonation',
        'coaching_target': 'intonation'
    },
    10: {
        'id': 10,
        'name': 'Natural Pausing',
        'group': 'prosody',
        'group_priority': 2,
        'activated_by': ['internal_long_pauses', 'fragmented_flow', 'excessive_internal_pauses'],
        'diagnostic_message': 'Pauses interrupt ideas instead of separating them',
        'coaching_target': 'natural_pausing'
    },

    # GROUP 3 — CONSONANT FLOW (Word-to-word continuity)
    3: {
        'id': 3,
        'name': 'Soft /b d g/',
        'group': 'consonants',
        'group_priority': 3,
        'activated_by': ['pauses_near_voiced_stops', 'flow_interruptions_between_words'],
        'diagnostic_message': 'Consonant transitions are too tense',
        'coaching_target': 'soft_bdg'
    },
    4: {
        'id': 4,
        'name': 'Calm /p t k/',
        'group': 'consonants',
        'group_priority': 3,
        'activated_by': ['pause_spikes_after_stops', 'flow_instability_after_voiceless'],
        'diagnostic_message': 'Consonant force is interrupting flow',
        'coaching_target': 'calm_ptk'
    },
    5: {
        'id': 5,
        'name': 'Mexican J /x/',
        'group': 'consonants',
        'group_priority': 3,
        'activated_by': ['pauses_before_friction', 'rate_drops_before_j'],
        'diagnostic_message': 'Friction sounds are blocking smooth airflow',
        'coaching_target': 'mexican_j'
    },
    6: {
        'id': 6,
        'name': 'Consonant Clusters',
        'group': 'consonants',
        'group_priority': 3,
        'activated_by': ['micro_pauses_in_clusters', 'fragmented_word_production'],
        'diagnostic_message': 'Complex words are being divided unnecessarily',
        'coaching_target': 'consonant_clusters'
    },

    # GROUP 4 — RHOTICS (Lower impact on comprehension)
    7: {
        'id': 7,
        'name': 'Tap /ɾ/',
        'group': 'rhotics',
        'group_priority': 4,
        'activated_by': ['pauses_in_function_words', 'rhythm_breaks_in_syllables'],
        'diagnostic_message': 'Short syllables lose connection',
        'coaching_target': 'tap_r'
    },
    8: {
        'id': 8,
        'name': 'Trill /r/',
        'group': 'rhotics',
        'group_priority': 4,
        'activated_by': ['preparation_pauses_before_trill', 'pre_articulatory_hesitation'],
        'diagnostic_message': 'Preparation interrupts natural rhythm',
        'coaching_target': 'trill_r'
    }
}

# Level band priority bias (what patterns to prioritize by level)
LEVEL_BAND_PRIORITIES = {
    'novice_low': ['fundamentals'],
    'novice_mid': ['fundamentals'],
    'novice_high': ['fundamentals'],
    'intermediate_low': ['fundamentals', 'prosody'],
    'intermediate_mid': ['fundamentals', 'prosody'],
    'intermediate_high': ['fundamentals', 'prosody'],
    'advanced_low': ['prosody', 'fundamentals'],
    'advanced_mid': ['prosody', 'consonants'],
    'advanced_high': ['prosody', 'consonants'],
    'superior': ['prosody', 'consonants'],
    'distinguished': ['prosody']  # refinement only
}

# Score-based pattern ceiling (max patterns to show)
SCORE_PATTERN_CEILING = {
    'under_60': {'max_patterns': 3, 'preferred_ids': [2, 1, 10]},
    '60_to_75': {'max_patterns': 3, 'preferred_groups': ['fundamentals', 'prosody']},
    '76_to_85': {'max_patterns': 2, 'preferred_groups': ['prosody', 'consonants']},
    'above_85': {'max_patterns': 1, 'preferred_groups': ['prosody']}  # optional refinement
}

def diagnose_patterns(signals, final_score, level='intermediate'):
    """CAPA 3 — Diagnostic Layer

    Translates evaluation signals from CAPA 2 into pedagogically meaningful patterns.

    CRITICAL RULES:
    - Patterns NEVER affect score (scoring is finalized before this runs)
    - Maximum 3 patterns
    - Priority hierarchy: Fundamentals > Prosody > Consonants > Rhotics
    - Level-sensitive prioritization

    Args:
        signals: List of signal tuples [(signal_name, severity), ...]
        final_score: Final score (0-100) from CAPA 2
        level: User-selected level (beginner/intermediate/advanced)

    Returns:
        List of activated patterns (max 3), sorted by priority
        [
            {
                'pattern_id': int,
                'priority': int,
                'name': str,
                'diagnostic_message': str,
                'coaching_target': str
            }
        ]
    """
    # STEP 1: Extract signal names
    signal_names = [sig[0] if isinstance(sig, tuple) else sig for sig in signals]

    # STEP 2: Identify which patterns are activated
    activated_patterns = []

    for pattern_id, pattern_def in DIAGNOSTIC_PATTERNS.items():
        # Check if any of this pattern's activation signals are present
        for activation_signal in pattern_def['activated_by']:
            if activation_signal in signal_names:
                activated_patterns.append({
                    'pattern_id': pattern_id,
                    'name': pattern_def['name'],
                    'group': pattern_def['group'],
                    'group_priority': pattern_def['group_priority'],
                    'diagnostic_message': pattern_def['diagnostic_message'],
                    'coaching_target': pattern_def['coaching_target']
                })
                break  # Only add once per pattern

    if not activated_patterns:
        return []  # No patterns to show

    # STEP 3: Get level band and preferred groups
    actfl_band = get_actfl_band(final_score)
    preferred_groups = LEVEL_BAND_PRIORITIES.get(actfl_band, ['fundamentals'])

    # STEP 4: Apply score-based ceiling
    if final_score < 60:
        ceiling = SCORE_PATTERN_CEILING['under_60']
        max_patterns = ceiling['max_patterns']
        # For scores < 60, use fixed pattern IDs if available
        if 'preferred_ids' in ceiling:
            preferred_pattern_ids = ceiling['preferred_ids']
            # Filter to only activated patterns with preferred IDs
            priority_patterns = [p for p in activated_patterns if p['pattern_id'] in preferred_pattern_ids]
            if priority_patterns:
                return priority_patterns[:max_patterns]
    elif final_score < 76:
        ceiling = SCORE_PATTERN_CEILING['60_to_75']
        max_patterns = ceiling['max_patterns']
    elif final_score < 86:
        ceiling = SCORE_PATTERN_CEILING['76_to_85']
        max_patterns = ceiling['max_patterns']
    else:
        ceiling = SCORE_PATTERN_CEILING['above_85']
        max_patterns = ceiling['max_patterns']

    # STEP 5: Prioritize by group order
    # Sort by: group_priority (lower is higher), then pattern_id
    activated_patterns.sort(key=lambda x: (x['group_priority'], x['pattern_id']))

    # STEP 6: Filter by preferred groups for this level band
    filtered_patterns = []
    for group_name in preferred_groups:
        group_patterns = [p for p in activated_patterns if p['group'] == group_name]
        filtered_patterns.extend(group_patterns)
        if len(filtered_patterns) >= max_patterns:
            break

    # If we don't have enough from preferred groups, fill with others
    if len(filtered_patterns) < max_patterns:
        remaining = [p for p in activated_patterns if p not in filtered_patterns]
        filtered_patterns.extend(remaining[:max_patterns - len(filtered_patterns)])

    # STEP 7: Return top N patterns with priority assigned
    final_patterns = []
    for idx, pattern in enumerate(filtered_patterns[:max_patterns]):
        final_patterns.append({
            'pattern_id': pattern['pattern_id'],
            'priority': idx + 1,  # 1 = highest
            'name': pattern['name'],
            'diagnostic_message': pattern['diagnostic_message'],
            'coaching_target': pattern['coaching_target']
        })

    return final_patterns

def evaluate_speech_clarity(transcript, words_data):
    """C1: Speech Clarity (25% weight)

    FACT Spec Section 2: Measures whether a listener can understand the spoken
    message without significant effort.

    Subcomponents (per spec):
    - C1.1 Overall Intelligibility (30%): STT confidence as ceiling, not penalty
    - C1.2 Thought Grouping (25%): Thinking pauses vs disruptive pauses
    - C1.3 Flow Continuity (25%): Sustained forward movement without interruptions
    - C1.4 Stability Over Time (20%): Consistent speaking rate (WPS std dev)

    Formula: C1 = (C1.1 × 0.30) + (C1.2 × 0.25) + (C1.3 × 0.25) + (C1.4 × 0.20)

    Returns:
        dict with 'score' (0-100), 'subcriteria', 'details'
    """
    if not words_data or len(words_data) == 0:
        return {
            'score': 70,
            'subcriteria': {
                'c1_1_intelligibility': 70,
                'c1_2_thought_grouping': 70,
                'c1_3_flow_continuity': 70,
                'c1_4_stability': 70
            },
            'details': {'note': 'No timing data available'}
        }

    # ===== C1.1: OVERALL INTELLIGIBILITY (30%) =====
    # FACT Spec 2.3: STT confidence is a CEILING, not a penalty
    # "Low confidence does NOT directly lower the score. It only prevents unrealistic inflation."
    try:
        confidences = [w.get('confidence', 0.75) for w in words_data if 'confidence' in w]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_confidence = 0.75

        # Base intelligibility score: If STT produced a transcription, speech was intelligible
        # Start with high base score (successful transcription = message understood)
        base_intelligibility = 95

        # Apply ceiling based on STT confidence (spec section 2.3)
        # This CAPS the score, it doesn't SET it
        if avg_confidence >= 0.85:
            ceiling = 100  # No cap applied
        elif avg_confidence >= 0.75:
            ceiling = 90   # Maximum 90 per spec
        elif avg_confidence >= 0.65:
            ceiling = 80   # Maximum 80 per spec
        else:
            ceiling = 70   # Maximum 70 per spec

        # Apply ceiling to base score
        c1_1_intelligibility = min(base_intelligibility, ceiling)
    except (KeyError, IndexError, TypeError, ZeroDivisionError) as e:
        logger.warning(f"Error calculating intelligibility: {e}")
        avg_confidence = 0.75
        c1_1_intelligibility = 90  # Default to ceiling for 0.75 confidence

    # ===== C1.2: THOUGHT GROUPING (25%) =====
    # Thinking pauses (between ideas) vs disruptive pauses (within phrases)
    # Per spec: "Thinking Pause: No penalty"
    try:
        thinking_pauses = 0
        disruptive_pauses = 0

        for i in range(len(words_data) - 1):
            gap = words_data[i+1]['start_time'] - words_data[i]['end_time']

            if gap >= 1.2:  # Pause threshold from spec
                current_word = words_data[i]['word'].lower()
                next_word = words_data[i+1]['word'].lower()

                # Expanded thinking pause markers: sentence boundaries and discourse markers
                # Natural speech includes pauses after connectors and before new ideas
                thinking_markers = ['.', '!', '?', ',', 'y', 'o',
                                   'entonces', 'luego', 'finalmente', 'después',
                                   'además', 'pero', 'sin embargo', 'porque',
                                   'bueno', 'pues', 'este', 'así', 'que',
                                   'primero', 'segundo', 'también', 'ahora']

                is_thinking_pause = any(marker in current_word for marker in thinking_markers)

                if is_thinking_pause:
                    thinking_pauses += 1
                else:
                    disruptive_pauses += 1

        # Score based on spec section 2.4 - adjusted for natural speech
        # Native speakers naturally pause; only penalize truly disruptive patterns
        if disruptive_pauses == 0:
            c1_2_thought_grouping = 95
        elif disruptive_pauses <= 2:
            c1_2_thought_grouping = 90
        elif disruptive_pauses <= 4:
            c1_2_thought_grouping = 80
        elif disruptive_pauses <= 6:
            c1_2_thought_grouping = 70
        else:
            c1_2_thought_grouping = 60
    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"Error calculating thought grouping: {e}")
        thinking_pauses = 0
        disruptive_pauses = 0
        c1_2_thought_grouping = 80

    # ===== C1.3: FLOW CONTINUITY (25%) =====
    # Sustained forward movement without unnecessary interruptions
    # Spec Section 2.5: Measures smooth, connected speech
    try:
        # Calculate total speech time vs total elapsed time
        if len(words_data) >= 2:
            total_elapsed = words_data[-1]['end_time'] - words_data[0]['start_time']
            total_speech = sum([w['end_time'] - w['start_time'] for w in words_data])

            speech_ratio = total_speech / total_elapsed if total_elapsed > 0 else 0

            # Count micro-pauses (0.3-1.2s) within phrases
            micro_pauses = 0
            for i in range(len(words_data) - 1):
                gap = words_data[i+1]['start_time'] - words_data[i]['end_time']
                if 0.3 <= gap < 1.2:
                    micro_pauses += 1

            # Score based on spec section 2.5 - adjusted for natural speech patterns
            # Natural spontaneous speech has more pauses than read speech
            if speech_ratio >= 0.65 and micro_pauses <= 4:
                c1_3_flow_continuity = 95  # Smooth, connected speech
            elif speech_ratio >= 0.55 and micro_pauses <= 6:
                c1_3_flow_continuity = 85  # Occasional interruptions (high end)
            elif speech_ratio >= 0.45 and micro_pauses <= 8:
                c1_3_flow_continuity = 75  # Occasional interruptions (low end)
            elif speech_ratio >= 0.35:
                c1_3_flow_continuity = 65  # Frequent fragmentation
            else:
                c1_3_flow_continuity = 55  # Severe breakdown
        else:
            c1_3_flow_continuity = 80
            speech_ratio = 0
            micro_pauses = 0
    except (KeyError, IndexError, TypeError, ZeroDivisionError) as e:
        logger.warning(f"Error calculating flow continuity: {e}")
        c1_3_flow_continuity = 80
        speech_ratio = 0
        micro_pauses = 0

    # ===== C1.4: STABILITY OVER TIME (20%) =====
    # Consistent speaking rate (WPS standard deviation)
    try:
        duration = words_data[-1]['end_time'] - words_data[0]['start_time']
        if duration > 3:
            window_wps = []
            window_size = 3.0
            start_time = words_data[0]['start_time']

            for i in range(int(duration // window_size)):
                window_start = start_time + i * window_size
                window_end = window_start + window_size
                words_in_window = [
                    w for w in words_data
                    if window_start <= w['start_time'] < window_end
                ]
                if words_in_window:
                    wps = len(words_in_window) / window_size
                    window_wps.append(wps)

            if len(window_wps) > 1:
                wps_std_dev = statistics.stdev(window_wps)

                # Score based on spec section 2.6 - adjusted for natural speech
                # Natural speech varies in pace; strict thresholds penalize spontaneity
                if wps_std_dev <= 0.40:
                    c1_4_stability = 95  # Stable rhythm (90-100 range)
                elif wps_std_dev <= 0.70:
                    c1_4_stability = 85  # Natural variation (75-89 range)
                elif wps_std_dev <= 1.0:
                    c1_4_stability = 70  # Irregular rhythm (60-74 range)
                else:
                    c1_4_stability = 55  # Collapsed rhythm (<60 range)
            else:
                c1_4_stability = 85  # Assume stable if not enough data
                wps_std_dev = 0
        else:
            c1_4_stability = 85  # Assume stable for short recordings
            wps_std_dev = 0
    except (KeyError, IndexError, TypeError, ZeroDivisionError, statistics.StatisticsError) as e:
        logger.warning(f"Error calculating stability: {e}")
        c1_4_stability = 75
        wps_std_dev = 0

    # ===== CALCULATE C1 FINAL SCORE =====
    c1_final_score = (c1_1_intelligibility * 0.30 +
                      c1_2_thought_grouping * 0.25 +
                      c1_3_flow_continuity * 0.25 +
                      c1_4_stability * 0.20)

    # ===== MINIMUM SCORE PROTECTION (Spec Section 2.7) =====
    # Protects competent speakers from false negatives
    # Per spec Section 10.1: "Native speakers: 85%+ score 85 or higher"

    # Strong protection: High confidence + good rhythm = native-like speech
    if avg_confidence >= 0.85 and disruptive_pauses <= 2 and wps_std_dev <= 0.60:
        c1_final_score = max(c1_final_score, 85)
    # Standard protection: Moderate confidence + reasonable rhythm
    elif avg_confidence >= 0.75 and disruptive_pauses <= 3 and wps_std_dev <= 0.75:
        c1_final_score = max(c1_final_score, 80)

    return {
        'score': round(c1_final_score, 1),
        'subcriteria': {
            'c1_1_intelligibility': round(c1_1_intelligibility, 1),
            'c1_2_thought_grouping': round(c1_2_thought_grouping, 1),
            'c1_3_flow_continuity': round(c1_3_flow_continuity, 1),
            'c1_4_stability': round(c1_4_stability, 1)
        },
        'details': {
            'stt_confidence': round(avg_confidence, 3),
            'thinking_pauses': thinking_pauses,
            'disruptive_pauses': disruptive_pauses,
            'speech_ratio': round(speech_ratio, 2) if speech_ratio else 0,
            'micro_pauses': micro_pauses if 'micro_pauses' in locals() else 0,
            'wps_std_dev': round(wps_std_dev, 2) if wps_std_dev else 0
        }
    }


def evaluate_communicative_function(transcript, level='intermediate'):
    """C2: Communicative Function (30% weight)

    FACT Spec Section 3: Measures what the speaker can DO with Spanish.

    Core Principle: "Grammar is evidence of function, not the evaluation target."

    Subcomponents (per spec):
    - C2.1 Task Fulfillment (30%): Addresses the prompt purpose
    - C2.2 Functional Control (30%): Sustained use of intended function
    - C2.3 Function Range (20%): Breadth of communicative actions
    - C2.4 Meaning Precision (20%): Intended meaning conveyed clearly

    Formula: C2 = (C2.1 × 0.30) + (C2.2 × 0.30) + (C2.3 × 0.20) + (C2.4 × 0.20)

    Returns:
        dict with 'score' (0-100), 'subcriteria', 'details', 'can_dos_detected'
    """
    text_lower = transcript.lower()
    can_dos_detected = []
    structures_detected = {}

    # ===== DETECT GRAMMATICAL STRUCTURES (Evidence of Function) =====
    # Spec Section 3.5: Structures detected as signals of functional intent

    # Present tense ser/estar (identification, location, states)
    structures_detected['presente_ser_estar'] = len(re.findall(
        r'\b(soy|eres|es|somos|son|estoy|estás|está|estamos|están)\b', text_lower
    ))

    # Present tense regular verbs (habitual actions)
    structures_detected['presente_regular'] = len(re.findall(
        r'\b(hablo|hablas|habla|hablamos|hablan|como|comes|come|comemos|comen|vivo|vives|vive|vivimos|viven|trabajo|trabajas|trabaja|trabajamos|trabajan|estudio|estudias|estudia|estudiamos|estudian)\b',
        text_lower
    ))

    # Possessive adjectives (possession)
    structures_detected['posesivos'] = len(re.findall(
        r'\b(mi|mis|tu|tus|su|sus|nuestro|nuestra|nuestros|nuestras)\b', text_lower
    ))

    # Tener (possession/age)
    structures_detected['tener'] = len(re.findall(
        r'\b(tengo|tienes|tiene|tenemos|tienen)\b', text_lower
    ))

    # Ir a + infinitive (future intention)
    structures_detected['ir_a'] = len(re.findall(
        r'\b(voy a|vas a|va a|vamos a|van a)\b', text_lower
    ))

    # Estar + gerund (present progressive)
    structures_detected['estar_gerundio'] = len(re.findall(
        r'\b(estoy|estás|está|estamos|están)\s+\w+(ando|iendo)\b', text_lower
    ))

    # Gustar (preferences)
    structures_detected['gustar'] = len(re.findall(
        r'\b(me gusta|me gustan|te gusta|te gustan|le gusta|le gustan|nos gusta|nos gustan)\b',
        text_lower
    ))

    # Preterite (completed actions in past)
    structures_detected['preterite'] = len(re.findall(
        r'\b(fui|fue|fueron|hice|hizo|hicieron|comí|comió|comieron|hablé|habló|hablaron|trabajé|trabajó|trabajaron|desperté|despertó|despertaron|regresé|regresó|regresaron)\b',
        text_lower
    ))

    # Imperfect (descriptions/habitual past)
    structures_detected['imperfect'] = len(re.findall(
        r'\b(era|eras|éramos|eran|estaba|estabas|estábamos|estaban|tenía|tenías|teníamos|tenían|iba|ibas|íbamos|iban|hacía|hacías|hacíamos|hacían)\b',
        text_lower
    ))

    # Reflexive verbs (daily routine)
    structures_detected['reflexive'] = len(re.findall(
        r'\b(me|te|se|nos)\s+(despert[oóé]|ducho|duchó|visto|vistió|llamo|llamó|siento|sintió)\b',
        text_lower
    ))

    # Subjunctive (emotion/influence/doubt at advanced)
    structures_detected['subjunctive'] = len(re.findall(
        r'\b(sea|seas|seamos|sean|esté|estés|estemos|estén|tenga|tengas|tengamos|tengan|haya|hayas|hayamos|hayan|pueda|puedas|podamos|puedan|quiera|quieras|queramos|quieran|es importante que|me preocupa que|espero que|no creo que|para que|ojalá)\b',
        text_lower
    ))

    # Conditional (hypothetical)
    structures_detected['conditional'] = len(re.findall(
        r'\b(sería|serías|seríamos|serían|haría|harías|haríamos|harían|tendría|tendrías|tendríamos|tendrían|podría|podrías|podríamos|podrían|debería|deberías|deberíamos|deberían)\b',
        text_lower
    ))

    # Commands/imperatives (instructions)
    structures_detected['commands'] = len(re.findall(
        r'\b(habla|hable|come|coma|escribe|escriba|ve|vaya|haz|haga|ten|tenga)\b',
        text_lower
    ))

    # Por vs para (purpose)
    structures_detected['por_para'] = len(re.findall(
        r'\b(por|para)\b', text_lower
    ))

    # ===== GATING: MINIMUM STRUCTURE REQUIREMENT =====
    # ACTFL principle: "No puedes evaluar lo que no existe"
    # If no grammatical structures detected, cannot evaluate communicative function
    total_structures_detected = sum(structures_detected.values())
    function_gating_active = total_structures_detected == 0

    # ===== C2.1: TASK FULFILLMENT (30%) =====
    # Does the speaker address the prompt purpose?
    c2_1_task_fulfillment = 50

    if function_gating_active:
        # Gating: No structures detected, cannot fulfill task
        c2_1_task_fulfillment = 35
    else:
        if level == 'beginner':
            # Beginner prompt: "Introduce Yourself" - expect personal info
            personal_info_markers = structures_detected['presente_ser_estar'] + structures_detected['tener']
            if personal_info_markers >= 3:
                c2_1_task_fulfillment = 90
            elif personal_info_markers >= 2:
                c2_1_task_fulfillment = 75
            elif personal_info_markers >= 1:
                c2_1_task_fulfillment = 60

        elif level == 'intermediate':
            # Intermediate prompt: "Describe Your Day" - expect past narration
            past_markers = structures_detected['preterite'] + structures_detected['imperfect']
            if past_markers >= 5:
                c2_1_task_fulfillment = 90
            elif past_markers >= 3:
                c2_1_task_fulfillment = 75
            elif past_markers >= 1:
                c2_1_task_fulfillment = 60

        elif level == 'advanced':
            # Advanced prompt: "Technology and Education" - expect opinion/evaluation
            evaluative_markers = structures_detected['subjunctive'] + structures_detected['conditional']
            has_opinion_phrases = bool(re.search(
                r'\b(creo que|pienso que|considero que|me parece que|en mi opinión|es importante que|es necesario que|me preocupa que)\b',
                text_lower
            ))
            if has_opinion_phrases and evaluative_markers >= 2:
                c2_1_task_fulfillment = 95
            elif has_opinion_phrases and evaluative_markers >= 1:
                c2_1_task_fulfillment = 85
            elif has_opinion_phrases:
                c2_1_task_fulfillment = 70

    # ===== C2.2: FUNCTIONAL CONTROL (30%) =====
    # Sustained use of intended communicative function
    c2_2_functional_control = 50

    if function_gating_active:
        # Gating: No structures detected, cannot demonstrate control
        c2_2_functional_control = 35
    else:
        if level == 'beginner':
            # Control over present tense description
            total_present = structures_detected['presente_ser_estar'] + structures_detected['presente_regular']
            if total_present >= 5:
                c2_2_functional_control = 90
            elif total_present >= 3:
                c2_2_functional_control = 75
            elif total_present >= 2:
                c2_2_functional_control = 60

        elif level == 'intermediate':
            # Control over narration (preterite + imperfect coordination)
            has_both = structures_detected['preterite'] >= 2 and structures_detected['imperfect'] >= 1
            total_past = structures_detected['preterite'] + structures_detected['imperfect']
            if has_both and total_past >= 6:
                c2_2_functional_control = 95
            elif total_past >= 4:
                c2_2_functional_control = 80
            elif total_past >= 2:
                c2_2_functional_control = 65

        elif level == 'advanced':
            # Control over argumentation (subjunctive + connectors + evaluative language)
            has_subjunctive = structures_detected['subjunctive'] >= 2
            has_conditional = structures_detected['conditional'] >= 1
            complex_structures = has_subjunctive or has_conditional
            if complex_structures:
                c2_2_functional_control = 85
            else:
                c2_2_functional_control = 65

    # ===== C2.3: FUNCTION RANGE (20%) =====
    # Breadth of communicative actions demonstrated
    c2_3_function_range = 50

    if function_gating_active:
        # Gating: No structures detected, cannot demonstrate range
        c2_3_function_range = 35
    else:
        # Count distinct function types used
        function_types_used = 0
        if structures_detected['presente_ser_estar'] >= 1: function_types_used += 1
        if structures_detected['preterite'] >= 1: function_types_used += 1
        if structures_detected['imperfect'] >= 1: function_types_used += 1
        if structures_detected['ir_a'] >= 1: function_types_used += 1
        if structures_detected['gustar'] >= 1: function_types_used += 1
        if structures_detected['subjunctive'] >= 1: function_types_used += 1
        if structures_detected['conditional'] >= 1: function_types_used += 1
        if structures_detected['reflexive'] >= 1: function_types_used += 1

        # Score based on variety
        if function_types_used >= 5:
            c2_3_function_range = 95
        elif function_types_used >= 4:
            c2_3_function_range = 85
        elif function_types_used >= 3:
            c2_3_function_range = 75
        elif function_types_used >= 2:
            c2_3_function_range = 65

    # ===== C2.4: MEANING PRECISION (20%) =====
    # Intended meaning conveyed without confusion
    c2_4_meaning_precision = 50

    if function_gating_active:
        # Gating: No structures detected, cannot demonstrate precision
        c2_4_meaning_precision = 35
    else:
        # Check for coherent use of structures (not random)
        # Higher precision = consistent tense use, appropriate modality
        word_count = len(transcript.split())
        structure_density = sum(structures_detected.values()) / max(word_count, 1)

        # Good precision: 0.2-0.4 structures per word (coherent functional language)
        if 0.2 <= structure_density <= 0.4:
            c2_4_meaning_precision = 90
        elif 0.15 <= structure_density <= 0.5:
            c2_4_meaning_precision = 80
        elif 0.10 <= structure_density:
            c2_4_meaning_precision = 70
        else:
            c2_4_meaning_precision = 60

    # ===== CALCULATE C2 FINAL SCORE =====
    c2_final_score = (c2_1_task_fulfillment * 0.30 +
                      c2_2_functional_control * 0.30 +
                      c2_3_function_range * 0.20 +
                      c2_4_meaning_precision * 0.20)

    return {
        'score': round(c2_final_score, 1),
        'subcriteria': {
            'c2_1_task_fulfillment': round(c2_1_task_fulfillment, 1),
            'c2_2_functional_control': round(c2_2_functional_control, 1),
            'c2_3_function_range': round(c2_3_function_range, 1),
            'c2_4_meaning_precision': round(c2_4_meaning_precision, 1)
        },
        'details': {
            'structures_detected': structures_detected,
            'function_types_used': function_types_used if 'function_types_used' in locals() else 0,
            'structure_density': round(structure_density, 3) if 'structure_density' in locals() else 0
        },
        'can_dos_detected': can_dos_detected
    }


def evaluate_discourse_organization(transcript, words_data=None):
    """C3: Discourse Organization (20% weight)

    FACT Spec Section 4: Measures how ideas are structured and connected.

    Core Principle: "Organization is about the listener's experience, not linguistic perfection."

    Subcomponents (per spec):
    - C3.1 Logical Sequencing (30%): Ideas follow predictable order
    - C3.2 Cohesion (30%): Ideas are linked meaningfully
    - C3.3 Development (20%): Ideas expanded beyond phrases
    - C3.4 Discourse Type Alignment (20%): Structure matches task type

    Formula: C3 = (C3.1 × 0.30) + (C3.2 × 0.30) + (C3.3 × 0.20) + (C3.4 × 0.20)

    Returns:
        dict with 'score' (0-100), 'subcriteria', 'details'
    """
    text_lower = transcript.lower()
    words = transcript.split()
    word_count = len(words)

    # ===== GATING: MINIMUM WORD COUNT FOR DISCOURSE EVALUATION =====
    # ACTFL principle: "No puedes evaluar lo que no existe"
    # Cannot evaluate discourse organization with very short utterances
    discourse_gating_active = word_count < 12

    # ===== DETECT CONNECTORS BY TYPE (Spec Section 4.4) =====
    # Beginner connectors
    beginner_connectors = {
        'additive': ['y', 'también'],
        'contrast': ['pero'],
        'sequence': ['primero', 'después', 'luego'],
        'temporal': ['ahora', 'hoy', 'mañana'],
        'causal': ['porque']
    }

    # Intermediate connectors
    intermediate_connectors = {
        'temporal_sequence': ['primero', 'después', 'luego', 'entonces', 'más tarde', 'finalmente'],
        'temporal_context': ['antes', 'mientras', 'durante', 'cuando'],
        'frequency': ['siempre', 'nunca', 'a veces', 'generalmente'],
        'comparison': ['más que', 'menos que', 'tan'],
        'causal': ['porque', 'por eso', 'entonces']
    }

    # Advanced connectors
    advanced_connectors = {
        'causal': ['porque', 'por eso', 'debido a', 'ya que', 'puesto que'],
        'purpose': ['para', 'para que', 'con el fin de'],
        'consequence': ['por lo tanto', 'entonces', 'así que', 'en consecuencia'],
        'concession': ['aunque', 'a pesar de', 'sin embargo'],
        'contrast': ['sin embargo', 'no obstante', 'por el contrario', 'en cambio'],
        'condition': ['si', 'en caso de', 'siempre que'],
        'projection': ['en el futuro', 'más adelante', 'eventualmente']
    }

    # Count connectors
    connector_counts = {}
    total_connectors = 0

    # Check all connector types (combine all levels)
    all_connectors = {**beginner_connectors, **intermediate_connectors, **advanced_connectors}
    for conn_type, conn_list in all_connectors.items():
        count = 0
        for connector in conn_list:
            count += text_lower.count(connector)
        connector_counts[conn_type] = count
        total_connectors += count

    connector_variety = sum(1 for count in connector_counts.values() if count > 0)

    # ===== C3.1: LOGICAL SEQUENCING (30%) =====
    # Ideas follow predictable order (temporal or logical)
    c3_1_logical_sequencing = 50

    if discourse_gating_active:
        # Gating: Too short to evaluate sequencing
        c3_1_logical_sequencing = 30
    else:
        temporal_sequencing = connector_counts.get('temporal_sequence', 0) + connector_counts.get('sequence', 0)
        logical_ordering = connector_counts.get('consequence', 0) + connector_counts.get('purpose', 0)

        total_sequencing = temporal_sequencing + logical_ordering

        if total_sequencing >= 4:
            c3_1_logical_sequencing = 95
        elif total_sequencing >= 3:
            c3_1_logical_sequencing = 85
        elif total_sequencing >= 2:
            c3_1_logical_sequencing = 75
        elif total_sequencing >= 1:
            c3_1_logical_sequencing = 65

    # ===== C3.2: COHESION (30%) =====
    # Ideas are linked meaningfully with functional connectors
    c3_2_cohesion = 50

    if discourse_gating_active:
        # Gating: Too short to evaluate cohesion
        c3_2_cohesion = 30
    else:
        cohesive_connectors = (connector_counts.get('causal', 0) +
                              connector_counts.get('contrast', 0) +
                              connector_counts.get('additive', 0) +
                              connector_counts.get('concession', 0))

        if cohesive_connectors >= 4 and connector_variety >= 3:
            c3_2_cohesion = 95
        elif cohesive_connectors >= 3 and connector_variety >= 2:
            c3_2_cohesion = 85
        elif cohesive_connectors >= 2:
            c3_2_cohesion = 75
        elif cohesive_connectors >= 1:
            c3_2_cohesion = 65

    # ===== C3.3: DEVELOPMENT (20%) =====
    # Ideas expanded beyond simple phrases
    c3_3_development = 50

    if discourse_gating_active:
        # Gating: Too short to evaluate idea development
        c3_3_development = 30
    else:
        # Count functional sentences (estimated by pause patterns or connectors)
        if words_data and len(words_data) > 0:
            functional_sentences = 1
            for i in range(len(words_data) - 1):
                gap = words_data[i+1]['start_time'] - words_data[i]['end_time']
                if gap >= 1.5:  # Strategic pause threshold
                    functional_sentences += 1
        else:
            # Fallback: estimate by connectors
            functional_sentences = max(1, total_connectors + 1)

        avg_idea_length = word_count / functional_sentences if functional_sentences > 0 else word_count

        # Score based on idea elaboration
        if avg_idea_length >= 15 and functional_sentences >= 3:
            c3_3_development = 95
        elif avg_idea_length >= 10 and functional_sentences >= 2:
            c3_3_development = 80
        elif avg_idea_length >= 7:
            c3_3_development = 70
        elif avg_idea_length >= 5:
            c3_3_development = 60

    # ===== C3.4: DISCOURSE TYPE ALIGNMENT (20%) =====
    # Structure matches the prompt task type
    c3_4_discourse_type = 50

    # Detect discourse type based on markers
    narrative_markers = ['ayer', 'primero', 'después', 'luego', 'entonces', 'finalmente', 'cuando']
    argumentative_markers = ['creo que', 'pienso que', 'considero que', 'me parece', 'es importante',
                            'me preocupa', 'aunque', 'sin embargo', 'por lo tanto']
    descriptive_markers = ['es', 'está', 'tiene', 'hay', 'son']

    narrative_count = sum(1 for marker in narrative_markers if marker in text_lower)
    argumentative_count = sum(1 for marker in argumentative_markers if marker in text_lower)
    descriptive_count = sum(1 for marker in descriptive_markers if marker in text_lower)

    # Determine discourse type
    if argumentative_count >= 3:
        discourse_type = 'argumentative'
        c3_4_discourse_type = 90
    elif narrative_count >= 3:
        discourse_type = 'narrative'
        c3_4_discourse_type = 90
    elif descriptive_count >= 3:
        discourse_type = 'descriptive'
        c3_4_discourse_type = 75
    else:
        discourse_type = 'conversational'
        c3_4_discourse_type = 60

    # ===== CALCULATE C3 FINAL SCORE =====
    c3_final_score = (c3_1_logical_sequencing * 0.30 +
                      c3_2_cohesion * 0.30 +
                      c3_3_development * 0.20 +
                      c3_4_discourse_type * 0.20)

    return {
        'score': round(c3_final_score, 1),
        'subcriteria': {
            'c3_1_logical_sequencing': round(c3_1_logical_sequencing, 1),
            'c3_2_cohesion': round(c3_2_cohesion, 1),
            'c3_3_development': round(c3_3_development, 1),
            'c3_4_discourse_type': round(c3_4_discourse_type, 1)
        },
        'details': {
            'word_count': word_count,
            'total_connectors': total_connectors,
            'connector_variety': connector_variety,
            'functional_sentences': functional_sentences if 'functional_sentences' in locals() else 0,
            'avg_idea_length': round(avg_idea_length, 1) if 'avg_idea_length' in locals() else 0,
            'discourse_type': discourse_type if 'discourse_type' in locals() else 'unknown',
            'connector_counts': connector_counts
        }
    }


def evaluate_lexical_use(transcript, level='intermediate'):
    """C4: Lexical Use (25% weight)

    FACT Spec Section 5: Measures how effectively vocabulary serves the communicative message.

    Core Principle: "Good vocabulary is functional vocabulary, not fancy vocabulary."

    Subcomponents (per spec):
    - C4.1 Lexical Fit (30%): Words match the topic
    - C4.2 Lexical Sufficiency (30%): Enough vocabulary to express ideas
    - C4.3 Lexical Variety (20%): Avoids excessive repetition
    - C4.4 Conceptual Level (20%): Vocabulary appropriate to task complexity

    Formula: C4 = (C4.1 × 0.30) + (C4.2 × 0.30) + (C4.3 × 0.20) + (C4.4 × 0.20)

    Returns:
        dict with 'score' (0-100), 'subcriteria', 'details'
    """
    text_lower = transcript.lower()
    words = transcript.lower().split()

    if not words:
        return {
            'score': 50,
            'subcriteria': {
                'c4_1_lexical_fit': 50,
                'c4_2_lexical_sufficiency': 50,
                'c4_3_lexical_variety': 50,
                'c4_4_conceptual_level': 50
            },
            'details': {}
        }

    # ===== EXPECTED VOCABULARY BY LEVEL (Spec Section 5.4) =====
    # Beginner Level Vocabulary
    beginner_vocab = {
        'identity': ['nombre', 'edad', 'nacionalidad', 'origen', 'llamo', 'años', 'soy'],
        'occupations': ['estudiante', 'profesor', 'trabajo', 'médico', 'estudio'],
        'languages': ['español', 'inglés', 'idioma', 'hablo', 'aprendiendo'],
        'family': ['familia', 'padre', 'madre', 'hermano', 'hijo', 'hermanos'],
        'hobbies': ['leer', 'cocinar', 'deportes', 'música', 'viajar', 'gusta'],
        'time': ['día', 'hora', 'mañana', 'tarde', 'noche']
    }

    # Intermediate Level Vocabulary
    intermediate_vocab = {
        'shopping': ['comprar', 'vender', 'precio', 'mercado', 'tienda', 'compré'],
        'daily_routine': ['despertar', 'desayunar', 'duchar', 'vestir', 'desperté', 'desayuné'],
        'food': ['comida', 'desayuno', 'almuerzo', 'cena', 'cocinar', 'café'],
        'health': ['salud', 'médico', 'enfermo', 'dolor', 'síntomas', 'dolía'],
        'past_activities': ['ayer', 'fui', 'hice', 'dije', 'comí', 'hablé', 'regresé'],
        'experiences': ['viaje', 'experiencia', 'evento', 'celebración', 'fue', 'sentía']
    }

    # Advanced Level Vocabulary
    advanced_vocab = {
        'technology': ['tecnología', 'digital', 'plataforma', 'herramienta', 'plataformas', 'digitales'],
        'education': ['educación', 'estudiante', 'aprendizaje', 'enseñanza', 'estudiantes', 'aprender'],
        'abstract_concepts': ['desarrollo', 'cambio', 'importancia', 'necesidad', 'importante', 'necesario'],
        'evaluation': ['beneficio', 'problema', 'desafío', 'ventaja', 'preocupa', 'parece'],
        'opinion_markers': ['creo', 'pienso', 'considero', 'opinión', 'perspectiva'],
        'emotion': ['preocupa', 'alegra', 'molesta', 'importa', 'emociona'],
        'future_projection': ['futuro', 'debería', 'podría', 'será', 'cambiar', 'híbrida']
    }

    # ===== COUNT TOPIC-ALIGNED KEYWORDS BY LEVEL =====
    topic_keywords_found = 0

    if level == 'beginner':
        for category, keywords in beginner_vocab.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topic_keywords_found += 1

    elif level == 'intermediate':
        for category, keywords in intermediate_vocab.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topic_keywords_found += 1

    elif level == 'advanced':
        for category, keywords in advanced_vocab.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topic_keywords_found += 1

    # ===== GATING: MINIMUM WORD COUNT REQUIREMENT =====
    # ACTFL principle: "No puedes evaluar lo que no existe"
    # If word count is too low, apply gating to prevent inflated scores
    word_count = len(words)

    # ===== C4.1: LEXICAL FIT (30%) =====
    # Words match the topic (Spec Section 5.5)
    c4_1_lexical_fit = 50

    if word_count < 10:
        # Gating: Insufficient words to demonstrate lexical fit
        c4_1_lexical_fit = 30
    elif word_count > 0:
        topic_alignment_ratio = topic_keywords_found / word_count

        if topic_alignment_ratio >= 0.30:  # 80%+ match (adjusted for shorter transcripts)
            c4_1_lexical_fit = 95
        elif topic_alignment_ratio >= 0.20:  # 60-79% match
            c4_1_lexical_fit = 80
        elif topic_alignment_ratio >= 0.10:  # 40-59% match
            c4_1_lexical_fit = 70
        elif topic_alignment_ratio >= 0.05:  # Some match
            c4_1_lexical_fit = 60

    # ===== C4.2: LEXICAL SUFFICIENCY (30%) =====
    # Enough vocabulary to express ideas
    c4_2_lexical_sufficiency = 50

    if word_count < 10:
        # Gating: Insufficient words to evaluate sufficiency
        c4_2_lexical_sufficiency = 25
    else:
        # Measure total keywords vs minimum needed
        if level == 'beginner':
            if topic_keywords_found >= 6:
                c4_2_lexical_sufficiency = 90
            elif topic_keywords_found >= 4:
                c4_2_lexical_sufficiency = 75
            elif topic_keywords_found >= 2:
                c4_2_lexical_sufficiency = 60

        elif level == 'intermediate':
            if topic_keywords_found >= 8:
                c4_2_lexical_sufficiency = 90
            elif topic_keywords_found >= 5:
                c4_2_lexical_sufficiency = 75
            elif topic_keywords_found >= 3:
                c4_2_lexical_sufficiency = 60

        elif level == 'advanced':
            if topic_keywords_found >= 10:
                c4_2_lexical_sufficiency = 95
            elif topic_keywords_found >= 7:
                c4_2_lexical_sufficiency = 80
            elif topic_keywords_found >= 4:
                c4_2_lexical_sufficiency = 65

    # ===== C4.3: LEXICAL VARIETY (20%) =====
    # Avoids excessive repetition
    c4_3_lexical_variety = 50

    if word_count < 10:
        # Gating: Cannot measure variety with <10 words
        c4_3_lexical_variety = 0
        variety_ratio = 0
    else:
        # Calculate variety ratio (unique words / total words)
        clean_words = [re.sub(r'[^\w\s]', '', w) for w in words if w]
        function_words = ['el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'a', 'al',
                         'en', 'con', 'por', 'para', 'que', 'y', 'o', 'pero', 'es', 'son', 'está', 'están']
        content_words = [w for w in clean_words if w and w not in function_words]

        if len(content_words) > 0:
            unique_content = set(content_words)
            variety_ratio = len(unique_content) / len(content_words)

            if variety_ratio >= 0.75:
                c4_3_lexical_variety = 95
            elif variety_ratio >= 0.65:
                c4_3_lexical_variety = 85
            elif variety_ratio >= 0.55:
                c4_3_lexical_variety = 75
            elif variety_ratio >= 0.45:
                c4_3_lexical_variety = 65
            else:
                c4_3_lexical_variety = 55
        else:
            variety_ratio = 0.5

    # ===== C4.4: CONCEPTUAL LEVEL (20%) =====
    # Vocabulary appropriate to task complexity
    c4_4_conceptual_level = 50

    # Detect thematic level based on vocabulary used
    personal_count = sum(1 for cat in beginner_vocab.values() for w in cat if w in text_lower)
    everyday_count = sum(1 for cat in intermediate_vocab.values() for w in cat if w in text_lower)
    abstract_count = sum(1 for cat in advanced_vocab.values() for w in cat if w in text_lower)

    # Score based on level-appropriate conceptual complexity
    if level == 'beginner':
        if personal_count >= 5:
            c4_4_conceptual_level = 90
            thematic_level = 'personal'
        elif personal_count >= 3:
            c4_4_conceptual_level = 75
            thematic_level = 'personal'
        else:
            c4_4_conceptual_level = 60
            thematic_level = 'basic'

    elif level == 'intermediate':
        if everyday_count >= 6:
            c4_4_conceptual_level = 90
            thematic_level = 'everyday'
        elif everyday_count >= 4:
            c4_4_conceptual_level = 75
            thematic_level = 'everyday'
        elif abstract_count >= 3:
            c4_4_conceptual_level = 95  # Bonus for exceeding level
            thematic_level = 'abstract'
        else:
            c4_4_conceptual_level = 60
            thematic_level = 'basic'

    elif level == 'advanced':
        if abstract_count >= 8:
            c4_4_conceptual_level = 95
            thematic_level = 'abstract'
        elif abstract_count >= 5:
            c4_4_conceptual_level = 85
            thematic_level = 'abstract'
        elif everyday_count >= 5:
            c4_4_conceptual_level = 70
            thematic_level = 'everyday'
        else:
            c4_4_conceptual_level = 60
            thematic_level = 'basic'

    # ===== CALCULATE C4 FINAL SCORE =====
    c4_final_score = (c4_1_lexical_fit * 0.30 +
                      c4_2_lexical_sufficiency * 0.30 +
                      c4_3_lexical_variety * 0.20 +
                      c4_4_conceptual_level * 0.20)

    return {
        'score': round(c4_final_score, 1),
        'subcriteria': {
            'c4_1_lexical_fit': round(c4_1_lexical_fit, 1),
            'c4_2_lexical_sufficiency': round(c4_2_lexical_sufficiency, 1),
            'c4_3_lexical_variety': round(c4_3_lexical_variety, 1),
            'c4_4_conceptual_level': round(c4_4_conceptual_level, 1)
        },
        'details': {
            'topic_keywords_found': topic_keywords_found,
            'variety_ratio': round(variety_ratio, 2) if 'variety_ratio' in locals() else 0,
            'unique_content_words': len(set(content_words)) if content_words else 0,
            'total_content_words': len(content_words) if content_words else 0,
            'thematic_level': thematic_level if 'thematic_level' in locals() else 'unknown'
        }
    }


def evaluate_prompt_alignment(transcript, c1_score, c2_score, c3_score, c4_score, prompt_type='free_speech'):
    """C5: Prompt Alignment (10% weight)

    Evaluates whether the student answered what was asked through:
    1. Checklist fulfillment - Covered expected points
    2. Global clarity - Average of C1-C4

    Key principle: Does NOT penalize creativity, only verifies minimal functional coverage.

    Args:
        transcript: Full transcribed text
        c1_score, c2_score, c3_score, c4_score: Scores from other criteria
        prompt_type: Type of prompt (free_speech or prompt-specific)

    Returns:
        dict with 'score' (0-100), 'details', and 'fulfillment_pct'
    """
    text_lower = transcript.lower()

    # --- CALCULATE GLOBAL CLARITY (average C1-C4) ---
    global_clarity = (c1_score + c2_score + c3_score + c4_score) / 4

    # --- CHECK PROMPT FULFILLMENT ---
    fulfillment_score = 85  # Default for free speech (no strict checklist)

    # For specific prompts, check if key elements are present
    if prompt_type == 'introduce_yourself':
        checklist_items = {
            'name_origin': any(marker in text_lower for marker in ['llamo', 'nombre', 'soy de', 'vengo de']),
            'age_occupation': any(marker in text_lower for marker in ['años', 'tengo', 'estudiante', 'trabajo']),
            'languages': any(marker in text_lower for marker in ['hablo', 'español', 'inglés', 'idioma']),
            'hobbies': any(marker in text_lower for marker in ['gusta', 'me encanta', 'interesa', 'hobby'])
        }
        covered = sum(1 for v in checklist_items.values() if v)
        fulfillment_pct = (covered / len(checklist_items)) * 100

        if fulfillment_pct == 100:
            fulfillment_score = 95
        elif fulfillment_pct >= 75:
            fulfillment_score = 85
        elif fulfillment_pct >= 50:
            fulfillment_score = 70
        else:
            fulfillment_score = 55

    elif prompt_type == 'describe_your_day':
        checklist_items = {
            'wake_time': any(marker in text_lower for marker in ['desperté', 'me levanté', 'hora', 'mañana']),
            'activities': any(marker in text_lower for marker in ['fui', 'hice', 'trabajé', 'comí', 'estudié']),
            'met_people': any(marker in text_lower for marker in ['amigo', 'familia', 'colega', 'hablé con', 'vi a']),
            'how_felt': any(marker in text_lower for marker in ['estaba', 'me sentí', 'contento', 'cansado', 'feliz'])
        }
        covered = sum(1 for v in checklist_items.values() if v)
        fulfillment_pct = (covered / len(checklist_items)) * 100

        if fulfillment_pct == 100:
            fulfillment_score = 95
        elif fulfillment_pct >= 75:
            fulfillment_score = 85
        elif fulfillment_pct >= 50:
            fulfillment_score = 70
        else:
            fulfillment_score = 55

    elif prompt_type == 'opinion_technology_education':
        checklist_items = {
            'positive_aspect': any(marker in text_lower for marker in ['importante', 'positivo', 'beneficio', 'útil', 'permite']),
            'concern': any(marker in text_lower for marker in ['preocupa', 'problema', 'desafío', 'dificultad', 'sin embargo']),
            'personal_experience': any(marker in text_lower for marker in ['mi experiencia', 'he usado', 'he aprendido', 'en mi caso']),
            'future_idea': any(marker in text_lower for marker in ['futuro', 'debería', 'necesario que', 'espero que', 'será'])
        }
        covered = sum(1 for v in checklist_items.values() if v)
        fulfillment_pct = (covered / len(checklist_items)) * 100

        if fulfillment_pct == 100:
            fulfillment_score = 95
        elif fulfillment_pct >= 75:
            fulfillment_score = 85
        elif fulfillment_pct >= 50:
            fulfillment_score = 70
        else:
            fulfillment_score = 55
    else:
        fulfillment_pct = 100  # Free speech, no specific requirements

    # --- CALCULATE C5 SCORE ---
    # Average of fulfillment and global clarity
    c5_score = (fulfillment_score + global_clarity) / 2

    return {
        'score': round(c5_score, 1),
        'details': {
            'fulfillment': round(fulfillment_score, 1),
            'global_clarity': round(global_clarity, 1),
            'fulfillment_pct': round(fulfillment_pct, 1) if prompt_type != 'free_speech' else 100
        }
    }


def actfl_fact_assessment(transcription_data, level='intermediate', prompt_type='free_speech'):
    """Main FACT Speech Evaluation System Assessment

    FACT Spec Section 7: Complete system implementation per spec version 1.0

    Weights (Spec Section 1.3):
    - C1: Speech Clarity: 25%
    - C2: Communicative Function: 30%
    - C3: Discourse Organization: 20%
    - C4: Lexical Use: 25%

    Formula (Spec Section 7.3):
    Final Score = (C1 × 0.25) + (C2 × 0.30) + (C3 × 0.20) + (C4 × 0.25)

    Args:
        transcription_data: dict with 'transcript' and 'words'
        level: Expected level (beginner/intermediate/advanced)
        prompt_type: Type of prompt (for logging only)

    Returns:
        dict with score, feedback, strengths, areas_for_improvement
    """
    transcript = transcription_data.get('transcript', '')
    words_data = transcription_data.get('words', [])

    # Error handling (Spec Section 7.4)
    if not transcript:
        return {
            'score': 50,
            'feedback': "Unable to process audio. Please try again.",
            'strengths': [],
            'areas_for_improvement': ["Audio quality affected scoring. Please try again in a quiet environment."],
            'fact_breakdown': {}
        }

    # ===== EVALUATE EACH CRITERION =====
    c1_speech_clarity = evaluate_speech_clarity(transcript, words_data)
    c2_communicative_function = evaluate_communicative_function(transcript, level=level)
    c3_discourse_organization = evaluate_discourse_organization(transcript, words_data=words_data)
    c4_lexical_use = evaluate_lexical_use(transcript, level=level)

    # ===== CALCULATE WEIGHTED FINAL SCORE (Spec Section 7.3) =====
    raw_score = (
        c1_speech_clarity['score'] * 0.25 +
        c2_communicative_function['score'] * 0.30 +
        c3_discourse_organization['score'] * 0.20 +
        c4_lexical_use['score'] * 0.25
    )

    # ===== FREE SPEECH ADJUSTMENT =====
    # Compensates for natural variation in spontaneous speech that automated
    # systems penalize but human listeners tolerate without effort.
    #
    # Justification:
    # 1. Free speech includes natural pauses, rhythm variations, and reformulations
    #    that machines penalize but humans filter out automatically
    # 2. STT confidence measures how well Google understood the audio, not speaker quality
    # 3. Research shows automated systems are systematically harsher than human raters
    #    for spontaneous speech due to penalizing natural disfluencies
    #
    # This adjustment bridges the gap between machine metrics and human comprehension.
    stt_confidence = c1_speech_clarity.get('details', {}).get('stt_confidence', 0.75)

    if stt_confidence >= 0.70:
        # Intelligible speech: +10 points adjustment
        free_speech_adjustment = 10
    elif stt_confidence >= 0.50:
        # Partially intelligible: +5 points adjustment
        free_speech_adjustment = 5
    else:
        # Low intelligibility: no adjustment
        free_speech_adjustment = 0

    final_score = min(100, raw_score + free_speech_adjustment)

    # ===== GENERATE FEEDBACK (Spec Section 9) =====
    feedback_text = _generate_score_explanation(final_score)
    strengths = _generate_strengths(final_score, c1_speech_clarity, c2_communicative_function,
                                    c3_discourse_organization, c4_lexical_use, level)
    improvements = _generate_improvements(final_score, c1_speech_clarity, c2_communicative_function,
                                         c3_discourse_organization, c4_lexical_use, level)

    logger.info(f"FACT Assessment (Level: {level}) - "
                f"C1: {c1_speech_clarity['score']}, "
                f"C2: {c2_communicative_function['score']}, "
                f"C3: {c3_discourse_organization['score']}, "
                f"C4: {c4_lexical_use['score']}, "
                f"Raw: {raw_score:.1f}, Adj: +{free_speech_adjustment}, "
                f"Final: {final_score:.1f}")

    return {
        'score': round(final_score, 1),
        'feedback': feedback_text,
        'strengths': strengths,  # 2-3 items, always provided (Spec Section 9.4)
        'areas_for_improvement': improvements,  # 1-2 items maximum (Spec Section 9.4)
        'fact_breakdown': {
            'speech_clarity': c1_speech_clarity['score'],
            'communicative_function': c2_communicative_function['score'],
            'discourse_organization': c3_discourse_organization['score'],
            'lexical_use': c4_lexical_use['score']
        },
        'subcriteria_breakdown': {
            'c1_subcriteria': c1_speech_clarity.get('subcriteria', {}),
            'c2_subcriteria': c2_communicative_function.get('subcriteria', {}),
            'c3_subcriteria': c3_discourse_organization.get('subcriteria', {}),
            'c4_subcriteria': c4_lexical_use.get('subcriteria', {})
        },
        'details': {
            'c1_details': c1_speech_clarity.get('details', {}),
            'c2_details': c2_communicative_function.get('details', {}),
            'c3_details': c3_discourse_organization.get('details', {}),
            'c4_details': c4_lexical_use.get('details', {})
        }
    }


def _generate_score_explanation(score):
    """Generate brief score explanation (Spec Section 9.1)

    Returns 1-2 sentence explanation based on score range.
    """
    if score >= 85:
        return "Your communication is clear and effective. You successfully express your ideas with minimal listener effort."
    elif score >= 75:
        return "Your message comes through clearly. Some aspects could be refined to improve overall clarity."
    elif score >= 65:
        return "You communicate your main ideas, though clarity and organization could be strengthened."
    elif score >= 55:
        return "Your message is partially clear. Focusing on key areas will help your communication improve."
    else:
        return "Your attempt shows effort. With practice in clarity and organization, your communication will strengthen."


def _generate_strengths(score, c1, c2, c3, c4, level):
    """Generate strengths list (Spec Section 9.2)

    Rule 1: Always provide at least 1 strength
    Rule 2: Maximum 3 strengths
    Rule 3: Balance strengths based on score

    Args:
        score: Final score
        c1, c2, c3, c4: Criterion results
        level: User proficiency level

    Returns:
        List of 1-3 strength strings
    """
    strengths = []

    # Speech Clarity strengths
    if c1['score'] >= 80:
        if c1.get('details', {}).get('wps_std_dev', 1.0) <= 0.35:
            strengths.append("Your speech flows naturally and is easy to follow.")
        else:
            strengths.append("Your ideas are grouped clearly with effective pauses.")
    elif c1['score'] >= 70:
        strengths.append("You maintain steady rhythm throughout your message.")

    # Communicative Function strengths
    if c2['score'] >= 85:
        strengths.append("Your communicative goal is clear and well-executed.")
    elif c2['score'] >= 75:
        strengths.append("You successfully accomplish what you set out to communicate.")

    # Discourse Organization strengths
    if c3['score'] >= 80:
        strengths.append("Your ideas follow a logical order.")
    elif c3['score'] >= 70:
        strengths.append("You connect your thoughts effectively.")

    # Lexical Use strengths
    if c4['score'] >= 80:
        strengths.append("Your vocabulary supports your message well.")
    elif c4['score'] >= 70:
        strengths.append("You use words that clearly express your ideas.")

    # Ensure at least 1 strength (Spec Rule 1)
    if not strengths:
        strengths.append("You're making an effort to communicate in Spanish.")

    # Limit to max 3 strengths (Spec Rule 2)
    return strengths[:3]


def _generate_improvements(score, c1, c2, c3, c4, level):
    """Generate improvements list based on score ranges and lowest-scoring criteria

    Maximum 2 improvements based on lowest-scoring criteria

    Args:
        score: Final score
        c1, c2, c3, c4: Criterion results
        level: User proficiency level

    Returns:
        List of 1-2 improvement strings
    """
    import random

    # Score-based feedback pools for each criterion
    IMPROVEMENT_FEEDBACK = {
        'c1': {  # Speech Clarity (Pronunciation, Rhythm, Flow)
            (0, 20): [
                "Work on recognizing and producing the most common sounds in Spanish. Your ears will get better at hearing the differences with practice.",
                "Try recording yourself saying individual words and comparing them to native speakers. Small steps build confidence!",
                "Focus on pronouncing vowel sounds clearly and consistently. Spanish vowels are your building blocks!"
            ],
            (21, 40): [
                "Practice linking short, familiar phrases together. Even simple combinations help build fluency.",
                "Try to maintain a steady pace when speaking, even if it's slow. Rhythm matters more than speed right now.",
                "Record yourself and listen back. Notice where you pause or hesitate, then practice those spots."
            ],
            (41, 54): [
                "You're making real progress! Now work on smoothing the transitions between words you already know.",
                "Pay attention to where native speakers pause for breath. Try to group your words into natural chunks instead of word-by-word.",
                "Practice asking and answering simple questions. This builds both vocabulary and confidence."
            ],
            (55, 64): [
                "Work on maintaining consistent stress patterns within words. Spanish has predictable stress rules that will help you sound more natural.",
                "Try shadowing (repeating immediately after) short audio clips from native speakers to improve your rhythm and intonation.",
                "Focus on connecting simple sentences smoothly. The flow between sentences is just as important as individual words."
            ],
            (65, 74): [
                "Practice using connectors like 'porque' (because), 'cuando' (when), and 'después' (after) to make your speech flow better.",
                "Focus on controlling your rhythm so it's steady throughout your message, not just in individual words.",
                "Pay attention to linking words together smoothly. In natural Spanish, words often flow into each other."
            ],
            (75, 84): [
                "Work on reducing repetition and filler words while maintaining natural speech flow.",
                "Pay attention to intonation patterns that convey emotion or emphasis, not just basic statement vs. question.",
                "Practice varying your sentence structure for more engaging speech."
            ],
            (85, 94): [
                "Refine subtle aspects of pronunciation like prosody (speech melody) that convey attitude and emotion.",
                "Work on controlling your speech rate strategically—slowing for emphasis, speeding for less important details.",
                "Pay attention to regional variations in pronunciation and choose which standard you want to model most closely."
            ],
            (95, 100): [
                "Focus on the most subtle aspects of native-like pronunciation, such as regional intonation patterns and emotional coloring.",
                "Explore literary and artistic uses of the language to deepen your appreciation of its full expressive potential.",
                "Practice using sophisticated rhetorical devices like metaphor or parallel structure when appropriate."
            ]
        },
        'c2': {  # Communicative Function (Grammar, Tenses, Structures)
            (0, 20): [
                "Focus on building your foundation with basic greetings and self-introductions. Practice saying simple phrases about yourself until they feel natural.",
                "Start with memorizing a few key phrases you use every day. Repetition is your friend at this stage!",
                "Practice simple present tense verbs for daily activities. Mastering 'yo soy' and 'yo tengo' is a great start."
            ],
            (21, 40): [
                "Practice forming simple complete sentences using subject-verb-object order.",
                "Work on using the present tense consistently when talking about your daily routine.",
                "Try describing things around you using simple adjectives. This expands your vocabulary naturally."
            ],
            (41, 54): [
                "Practice expressing simple ideas in complete sentences, even if they're basic.",
                "Work on using the present tense consistently to talk about your daily activities and preferences.",
                "Try explaining simple processes (like making a sandwich) to practice using sequential language naturally."
            ],
            (55, 64): [
                "Practice using the present tense consistently across different subjects (I, you, he/she, we, they).",
                "Work on asking questions in addition to making statements. Questions help keep conversations flowing.",
                "Try combining two simple sentences with 'y' (and) or 'pero' (but) for more natural speech."
            ],
            (65, 74): [
                "Practice switching between present and past tense to tell simple stories about what you did yesterday.",
                "Work on using common irregular verbs in the present tense until they become automatic.",
                "Try expressing your opinions with 'Me gusta...' and 'Prefiero...' to add personality to your speech."
            ],
            (75, 84): [
                "Practice narrating past events with consistent use of preterite and imperfect tenses. This will make your stories more engaging.",
                "Work on using the future tense or 'ir a + infinitive' to talk about plans and predictions.",
                "Try using the subjunctive mood after expressions of desire or doubt to add nuance to your communication."
            ],
            (85, 94): [
                "Refine your ability to express abstract ideas and opinions with supporting details.",
                "Practice using subjunctive mood consistently when expressing doubt, desire, or hypotheticals.",
                "Work on constructing complex arguments with multiple supporting points and conclusions."
            ],
            (95, 100): [
                "Continue refining your ability to discuss highly specialized or abstract topics with precision.",
                "Practice constructing sophisticated arguments with counterarguments and nuanced positions.",
                "Work on adapting your register (formal vs. informal) smoothly based on context and audience."
            ]
        },
        'c3': {  # Discourse Organization (Connectors, Coherence, Structure)
            (0, 20): [
                "Practice saying simple phrases in a logical order, like 'Hello, my name is... I am from...'",
                "Work on responding to basic questions with appropriate answers, even if they're very short.",
                "Try memorizing short dialogues to understand how conversations flow naturally."
            ],
            (21, 40): [
                "Practice putting your ideas in a clear order: first introduce yourself, then add details.",
                "Work on using 'y' (and) to connect related ideas together.",
                "Try answering 'why' questions with 'porque' (because) to show cause and effect."
            ],
            (41, 54): [
                "Practice organizing your thoughts before speaking. Think: What do I want to say first, second, third?",
                "Work on using time words like 'primero' (first), 'después' (after), and 'finalmente' (finally).",
                "Try telling a short story with a beginning, middle, and end."
            ],
            (55, 64): [
                "Practice using basic connectors like 'y' (and), 'pero' (but), 'porque' (because) to link your sentences.",
                "Work on giving examples to support your main ideas using 'por ejemplo' (for example).",
                "Try explaining your reasoning step-by-step rather than jumping between unrelated ideas."
            ],
            (65, 74): [
                "Practice using a variety of connectors: 'porque', 'pero', 'entonces', 'también', 'sin embargo'.",
                "Work on organizing longer responses with a clear beginning, middle, and end.",
                "Try using transition phrases like 'por otro lado' (on the other hand) to show contrasts."
            ],
            (75, 84): [
                "Practice using sophisticated connectors like 'sin embargo' (nevertheless), 'por lo tanto' (therefore) to show clearer relationships between ideas.",
                "Work on signaling topic changes explicitly so listeners can follow your train of thought.",
                "Try adding supporting details and examples to make your main points more convincing."
            ],
            (85, 94): [
                "Practice incorporating more idiomatic expressions and culturally specific references naturally.",
                "Work on using cohesion devices like pronouns and synonyms to avoid repetition while maintaining clarity.",
                "Try structuring complex arguments with thesis, supporting evidence, and conclusion."
            ],
            (95, 100): [
                "Continue exposing yourself to diverse registers and specialized discourse structures.",
                "Practice tailoring your organizational patterns to different audiences and purposes.",
                "Work on using rhetorical devices strategically to enhance your message's impact."
            ]
        },
        'c4': {  # Lexical Use (Vocabulary Breadth and Precision)
            (0, 20): [
                "Build your vocabulary by learning 5 new common words each day. Start with things you see around you.",
                "Practice the most frequent words in Spanish: numbers, colors, family members, and common objects.",
                "Use flashcards or a vocabulary app to review new words regularly until they stick."
            ],
            (21, 40): [
                "Practice using adjectives to describe nouns. Start simple: big/small, good/bad, new/old.",
                "Work on learning vocabulary in categories: foods, places, activities. This makes it easier to remember.",
                "Try labeling objects around your home with Spanish sticky notes to build everyday vocabulary."
            ],
            (41, 54): [
                "Practice using verbs beyond 'ser' and 'estar'. Learn common action verbs for daily activities.",
                "Work on building topic-specific vocabulary for things you talk about frequently.",
                "Try using a Spanish-Spanish dictionary to learn new words through definitions rather than translations."
            ],
            (55, 64): [
                "Practice expressing the same idea in different ways to avoid repeating the same words.",
                "Work on learning synonyms for common words you use frequently to add variety.",
                "Try reading simple Spanish texts and noting down new words with example sentences."
            ],
            (65, 74): [
                "Practice using more specific vocabulary rather than general words. Instead of 'cosa', use the precise term.",
                "Work on learning common idiomatic expressions that native speakers use naturally.",
                "Try incorporating new vocabulary immediately after learning it, even if it feels awkward at first."
            ],
            (75, 84): [
                "Practice using topic-specific vocabulary with precision rather than general approximations.",
                "Work on distinguishing between similar words with subtle meaning differences.",
                "Try learning word families (noun, verb, adjective forms) to expand vocabulary efficiently."
            ],
            (85, 94): [
                "Refine your use of low-frequency but precise vocabulary for specialized topics.",
                "Work on using collocations (word combinations) that native speakers use naturally.",
                "Try incorporating more sophisticated vocabulary while maintaining clarity and appropriateness."
            ],
            (95, 100): [
                "Continue expanding your vocabulary in specialized domains relevant to your interests.",
                "Practice using vocabulary with complete awareness of register, connotation, and context.",
                "Explore historical, regional, and stylistic variations in vocabulary use."
            ]
        }
    }

    def get_score_range(score_val):
        """Determine which score range a value falls into"""
        if score_val <= 20:
            return (0, 20)
        elif score_val <= 40:
            return (21, 40)
        elif score_val <= 54:
            return (41, 54)
        elif score_val <= 64:
            return (55, 64)
        elif score_val <= 74:
            return (65, 74)
        elif score_val <= 84:
            return (75, 84)
        elif score_val <= 94:
            return (85, 94)
        else:
            return (95, 100)

    improvements = []

    # Identify lowest-scoring criteria
    criteria_scores = [
        ('c1', c1['score'], c1),
        ('c2', c2['score'], c2),
        ('c3', c3['score'], c3),
        ('c4', c4['score'], c4)
    ]
    criteria_scores.sort(key=lambda x: x[1])

    # Generate improvements for 2 lowest criteria
    for criterion_name, criterion_score, criterion_data in criteria_scores[:2]:
        score_range = get_score_range(criterion_score)

        if criterion_name in IMPROVEMENT_FEEDBACK and score_range in IMPROVEMENT_FEEDBACK[criterion_name]:
            feedback_options = IMPROVEMENT_FEEDBACK[criterion_name][score_range]
            if feedback_options:
                # Randomly select one feedback from the options for variety
                selected_feedback = random.choice(feedback_options)
                improvements.append(selected_feedback)

    # Fallback if somehow no improvements were generated
    if not improvements:
        improvements.append("Continue practicing regularly. Consistent exposure and use of Spanish will help you improve across all areas.")

    return improvements[:2]  # Max 2 improvements


def _generate_rubric_feedback(score, level='intermediate'):
    """Generate feedback aligned with instructor's rubric language and level"""
    if score >= 85:
        return "Excellent work - you communicate clearly and confidently with strong control of Spanish structures."
    elif score >= 75:
        return "Good work - you communicate effectively with clear pronunciation and good structural control."
    elif score >= 60:
        return "You're making progress - continue practicing to improve fluency, clarity, and grammatical consistency."
    else:
        return "Keep practicing - focus on forming complete sentences, improving pronunciation clarity, and building vocabulary."


def _generate_rubric_strengths_v2(score, c1_pronunciation, c2_functions, c3_text_type, c4_context, level):
    """Generate specific strengths based on actual performance (NEW version for FACT system)"""
    strengths = []

    # C1: Pronunciation strengths
    if c1_pronunciation['score'] >= 85:
        details = c1_pronunciation.get('details', {})
        if details.get('rhythm_stability', 0) >= 85:
            strengths.append("Your pronunciation is clear with stable rhythm")
        if details.get('flow_continuity', 0) >= 85:
            strengths.append("You speak naturally with minimal hesitation")
    elif c1_pronunciation['score'] >= 70:
        strengths.append("Your pronunciation is generally clear and comprehensible")

    # C2: Functions strengths (with modality awareness)
    modality = c2_functions.get('modality_detected', {})
    detected = c2_functions.get('detected', {})

    if c2_functions['score'] >= 85:
        if modality.get('evaluative', 0) > 0:
            strengths.append("You express opinions and evaluations effectively")
        elif detected.get('subjunctive', 0) >= 2:
            strengths.append("You demonstrate control of advanced grammatical structures")
        elif modality.get('cognitive', 0) > 0:
            strengths.append("You express your thoughts and opinions clearly")
    elif c2_functions['score'] >= 70:
        strengths.append("You use appropriate grammatical structures for communication")

    # C3: Text Type strengths
    discourse_details = c3_text_type.get('details', {})
    discourse_type = discourse_details.get('discourse_type', 'conversational')

    if c3_text_type['score'] >= 80:
        if discourse_type in ['academic', 'argumentative']:
            strengths.append("You organize ideas in academic/argumentative discourse")
        else:
            strengths.append("You convey ideas in complete, connected sentences")
    elif c3_text_type['score'] >= 65:
        strengths.append("You speak in complete sentences")

    # C4: Context (Vocabulary) strengths
    thematic_level = c4_context.get('thematic_level', 'basic')

    if c4_context['score'] >= 85:
        if thematic_level == 'abstract':
            strengths.append("You demonstrate sophisticated vocabulary use")
        else:
            strengths.append("You demonstrate good variety in your vocabulary")

    # If no specific strengths identified, add a general one
    if not strengths:
        strengths.append("You're making an effort to communicate in Spanish")

    return strengths


def _generate_rubric_improvements_v2(score, c1_pronunciation, c2_functions, c3_text_type, c4_context, level):
    """Generate specific, actionable improvements (NEW version for FACT system)"""
    improvements = []

    # C1: Pronunciation improvements
    details = c1_pronunciation.get('details', {})

    if c1_pronunciation['score'] < 75:
        if details.get('internal_pauses', 0) > 2:
            improvements.append("Practice speaking more smoothly - pause between ideas, not within them")
        if details.get('wps_std_dev', 0) > 0.6:
            improvements.append("Work on maintaining a more consistent speaking rhythm")
        elif details.get('wps_collapses', 0) > 2:
            improvements.append("Keep vowels short and stable within words")

    # C2: Functions improvements (with modality awareness)
    modality = c2_functions.get('modality_detected', {})
    detected = c2_functions.get('detected', {})

    if level == 'beginner':
        if c2_functions['score'] < 75 and detected.get('present', 0) < 3:
            improvements.append("Practice using present tense to describe yourself and your routine")
        if modality.get('basic', 0) == 0:
            improvements.append("Try expressing preferences with 'me gusta' or 'quiero'")

    elif level == 'intermediate':
        if c2_functions['score'] < 75:
            if detected.get('preterite', 0) + detected.get('imperfect', 0) < 3:
                improvements.append("Practice using past tense to narrate completed actions and descriptions")
        if modality.get('cognitive', 0) == 0:
            improvements.append("Try expressing opinions with 'creo que' or 'pienso que'")

    elif level == 'advanced':
        if c2_functions['score'] < 85:
            if modality.get('evaluative', 0) == 0:
                improvements.append("Use evaluative expressions like 'es importante que' or 'me preocupa que'")
            if detected.get('subjunctive', 0) == 0:
                improvements.append("Challenge yourself with subjunctive mood to express doubt, emotion, or necessity")

    # C3: Text Type improvements
    discourse_details = c3_text_type.get('details', {})

    if c3_text_type['score'] < 70:
        if discourse_details.get('total_connectors', 0) < 2:
            improvements.append("Connect your ideas with words like 'porque', 'pero', 'entonces', 'sin embargo'")
        if discourse_details.get('functional_sentences', 0) < 3:
            improvements.append("Try speaking in multiple complete sentences instead of isolated phrases")

    # C4: Context improvements
    context_details = c4_context.get('details', {})
    thematic_level = c4_context.get('thematic_level', 'basic')

    if c4_context['score'] < 70:
        if context_details.get('variety_ratio', 0) < 0.60:
            improvements.append("Expand your vocabulary to avoid repeating the same words")

    # General improvement if specific ones don't apply
    if not improvements:
        improvements.append("Continue refining subtle aspects of pronunciation and discourse organization")

    return improvements


# =============================================================================
# WRAPPER FUNCTIONS - Simplified to use FACT assessment
# =============================================================================

def assess_free_speech(transcription_data, level='intermediate'):
    """Evaluate free speech using FACT assessment

    Args:
        transcription_data: dict with 'transcript' and 'words'
        level: Expected proficiency level (beginner/intermediate/advanced)

    Returns:
        FACT assessment result
    """
    return actfl_fact_assessment(transcription_data, level=level, prompt_type='free_speech')

def assess_practice_phrase(transcription_data, reference_level, level='intermediate'):
    """Evaluate practice phrase using FACT assessment + similarity bonus

    Args:
        transcription_data: dict with 'transcript' and 'words'
        reference_level: Level key for reference phrase (short/medium/extended)
        level: Expected proficiency level (beginner/intermediate/advanced)

    Returns:
        FACT assessment result with similarity bonus and reference-specific feedback
    """
    transcript = transcription_data.get('transcript', '')

    if reference_level not in REFERENCES:
        return actfl_fact_assessment(transcription_data, level=level, prompt_type='free_speech')

    reference_text = REFERENCES[reference_level]

    # Map reference level to prompt type for better alignment checking
    prompt_type_map = {
        'short': 'introduce_yourself',
        'medium': 'describe_your_day',
        'extended': 'opinion_technology_education'
    }
    prompt_type = prompt_type_map.get(reference_level, 'free_speech')

    # Get base FACT assessment with appropriate prompt type
    base_assessment = actfl_fact_assessment(transcription_data, level=level, prompt_type=prompt_type)

    # Calculate similarity to reference phrase
    similarity_score = fuzz.token_sort_ratio(transcript.lower(), reference_text.lower())

    # Small bonus for following the reference (max +10 points)
    similarity_bonus = (similarity_score - 70) * 0.2 if similarity_score > 70 else 0
    adjusted_score = min(100, base_assessment['score'] + similarity_bonus)

    # Add reference-specific feedback
    if similarity_score < 60:
        base_assessment['areas_for_improvement'].insert(0,
            "Try to follow the reference phrase more closely")
    elif similarity_score >= 85:
        base_assessment['strengths'].insert(0,
            "Excellent reproduction of the reference phrase")

    # Update score and add reference data
    base_assessment['score'] = round(adjusted_score, 1)
    base_assessment['reference_text'] = reference_text
    base_assessment['reference_similarity'] = similarity_score

    return base_assessment

# Calculate pronunciation score based on clarity metrics
def pronunciation_assessment(transcribed_text):
    """
    Evaluate pronunciation using clarity metrics:
    - Accuracy: How precise is their pronunciation?
    - Recognition: How well are words recognized?
    - Text complexity: Can they produce appropriate sentence structures?
    - Vocabulary variety: Range of vocabulary used
    """
    words = transcribed_text.split()
    if not words:
        logger.warning("No words to score")
        return {
            "score": 70.0,
            "level": "Profile B - Functional Clarity",
            "feedback": "We couldn't detect your speech. Please ensure your microphone is working and try speaking a bit louder.",
            "strengths": [],
            "areas_for_improvement": ["Check microphone connection and reduce background noise"]
        }
    
    # Score each word's pronunciation accuracy
    word_scores = []
    mispronounced_words = []
    
    for word in words:
        word = word.lower()
        if word in SPANISH_DICT:
            score = 100  # Perfect match
            logger.info(f"Word '{word}' found in dictionary, score: 100")
        else:
            # Find best match using fuzzy matching
            best_match = None
            best_ratio = 0
            
            # Check against a random sample of the dictionary for performance (unbiased)
            if len(SPANISH_DICT) > 1000:
                dict_sample = set(random.sample(list(SPANISH_DICT), 1000))
            else:
                dict_sample = SPANISH_DICT
            
            for dict_word in dict_sample:
                ratio = fuzz.ratio(word, dict_word)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = dict_word
            
            score = best_ratio
            logger.info(f"Word '{word}' not found. Best match: '{best_match}' with score: {score}")
            
            if score < 80:
                mispronounced_words.append(word)
        
        word_scores.append(score)
    
    # Calculate average accuracy score
    accuracy_score = sum(word_scores) / len(word_scores) if word_scores else 70
    
    # Evaluate overall proficiency criteria
    
    # 1. Functions and tasks: Based on successfully recognized words
    recognized_word_percentage = sum(1 for s in word_scores if s >= 80) / len(words)
    
    # 2. Text type: Based on number of words (complexity of response)
    text_complexity = min(100, 60 + len(words) * 4) # Bonus for longer phrases
    
    # 3. Content: Based on variety of vocabulary (unique words ratio)
    unique_words_ratio = len(set(words)) / len(words)
    vocabulary_score = min(100, unique_words_ratio * 100)
    
    # Calculate composite clarity score with different weights
    # Pronunciation accuracy is most important
    composite_score = (
        accuracy_score * 0.6 +
        recognized_word_percentage * 100 * 0.2 +
        text_complexity * 0.1 +
        vocabulary_score * 0.1
    )

    # High clarity adjustment - boost scores for very clear pronunciation
    if accuracy_score > 90 and len(words) > 3:
        composite_score = min(100, composite_score + 5)

    # Determine pronunciation profile
    level = determine_pronunciation_profile(composite_score, len(words), recognized_word_percentage)
    
    # Generate feedback
    strengths = generate_strengths(accuracy_score, recognized_word_percentage, len(words))
    areas_for_improvement = generate_improvements(mispronounced_words, accuracy_score)
    
    logger.info(f"Clarity Scoring - Accuracy: {accuracy_score}, Recognition: {recognized_word_percentage*100}, " +
              f"Text: {text_complexity}, Vocab: {vocabulary_score}, Final: {composite_score}, Profile: {level}")

    # Build the assessment result
    assessment_result = {
        "score": round(composite_score, 1),
        "level": level,
        "feedback": generate_feedback(level),
        "strengths": strengths,
        "areas_for_improvement": areas_for_improvement,
        "word_scores": dict(zip(words, word_scores))
    }

    return assessment_result

def determine_pronunciation_profile(score, word_count, recognized_ratio):
    """Determine pronunciation profile based on clarity score and performance factors"""

    # If we have loaded profile criteria, use the score ranges from there
    if PROFILE_CRITERIA:
        # Adjust score based on performance factors
        # Baseline +5 bonus removed - scores now reflect actual evidence
        adjusted_score = score

        # Boost score for longer, more complex speech
        if word_count >= 15 and recognized_ratio >= 0.9:
            adjusted_score = min(100, score + 3)
        elif word_count >= 10 and recognized_ratio >= 0.85:
            adjusted_score = min(100, score + 2)
        elif word_count >= 5 and recognized_ratio >= 0.7:
            adjusted_score = min(100, score + 1)

        # Penalize for very short utterances or low recognition
        if word_count < 3 or recognized_ratio < 0.5:
            adjusted_score = max(0, score - 5)

        # Find the matching profile based on score range
        for profile_key, criteria in PROFILE_CRITERIA.items():
            score_min, score_max = criteria["score_range"]
            if score_min <= adjusted_score <= score_max:
                return criteria["name"]

        # Fallback to Profile A if no match found
        return "Profile A - Initial Clarity"

    # Fallback to built-in logic if criteria file not loaded
    # Profile C — Consistent Clarity (85-100)
    if score >= 85:
        if score >= 96:
            return "Profile C - Consistent Clarity (High)"
        elif score >= 91:
            return "Profile C - Consistent Clarity (Mid)"
        else:
            return "Profile C - Consistent Clarity (Low)"

    # Profile B — Functional Clarity (65-84)
    elif score >= 65:
        if score >= 78:
            return "Profile B - Functional Clarity (High)"
        elif score >= 72:
            return "Profile B - Functional Clarity (Mid)"
        else:
            return "Profile B - Functional Clarity (Low)"

    # Profile A — Initial Clarity (0-64)
    else:
        if score >= 43:
            return "Profile A - Initial Clarity (High)"
        elif score >= 21:
            return "Profile A - Initial Clarity (Mid)"
        else:
            return "Profile A - Initial Clarity (Low)"

def generate_feedback(level):
    """Generate feedback text based on pronunciation profile"""

    # If we have loaded profile criteria, use feedback templates from there
    if PROFILE_CRITERIA:
        for profile_key, criteria in PROFILE_CRITERIA.items():
            if criteria["name"] == level:
                return criteria["feedback_template"]

    # Fallback to built-in feedback templates based on profile ranges
    # Profile C — Consistent Clarity (85-100)
    if "Profile C" in level:
        return "Your pronunciation is clear and generally consistent. Small refinements will help improve overall naturalness and ease of understanding."

    # Profile B — Functional Clarity (65-84)
    elif "Profile B" in level:
        return "Your pronunciation supports functional communication, with some areas that would benefit from greater consistency."

    # Profile A — Initial Clarity (0-64)
    else:
        return "Your pronunciation is still developing. Keep going — consistency builds clarity."

def generate_strengths(accuracy, recognition_ratio, word_count):
    """Generate list of strengths based on performance profile - exactly 3 bullets"""
    strengths = []

    # Determine which profile we're in based on accuracy score (primary indicator)
    # This ensures feedback aligns with the clarity range

    # Profile C — Consistent Clarity (85-100)
    if accuracy >= 85:
        strengths.append("Most individual sounds were produced clearly.")
        strengths.append("Your words were consistently recognized, supporting strong intelligibility.")
        strengths.append("You maintained a steady rhythm across connected speech.")

    # Profile B — Functional Clarity (65-84)
    elif accuracy >= 65:
        strengths.append("Most words were pronounced clearly.")
        strengths.append("Your speech was generally easy for the system to recognize.")
        strengths.append("You maintained a steady rhythm across familiar phrases.")

    # Profile A — Initial Clarity (0-64)
    else:
        strengths.append("Some individual sounds were produced clearly.")
        strengths.append("Parts of your speech were recognized by the system, supporting partial understanding.")
        strengths.append("You showed emerging control over rhythm in familiar phrases.")

    return strengths

def generate_improvements(mispronounced, accuracy):
    """Generate suggested areas for improvement based on profile"""
    improvements = []

    # Profile-specific improvement suggestion (first bullet)
    # Profile C — Consistent Clarity (85-100)
    if accuracy >= 85:
        improvements.append("Focus on subtle refinements to rhythm and sound precision.")

    # Profile B — Functional Clarity (65-84)
    elif accuracy >= 65:
        improvements.append("Continue stabilizing rhythm and sound consistency in longer phrases.")

    # Profile A — Initial Clarity (0-64)
    else:
        improvements.append("Work on maintaining a smoother, more even rhythm while speaking.")

    # Specific Mispronunciation Feedback - SINGLE INTEGRATED SENTENCE (second bullet)
    if mispronounced:
        # Helper function to convert digits to Spanish words
        def number_to_spanish(word):
            """Convert digit strings to Spanish words"""
            digit_map = {
                '0': 'cero', '1': 'uno', '2': 'dos', '3': 'tres', '4': 'cuatro',
                '5': 'cinco', '6': 'seis', '7': 'siete', '8': 'ocho', '9': 'nueve',
                '10': 'diez'
            }
            return digit_map.get(word, word)

        # Convert numbers to Spanish and format words (limit to 5 words)
        formatted_words = [number_to_spanish(word) for word in mispronounced[:5]]

        # Build the sentence with proper formatting (using HTML for bold)
        if len(formatted_words) == 1:
            word_list = f"<strong>{formatted_words[0]}</strong>"
        elif len(formatted_words) == 2:
            word_list = f"<strong>{formatted_words[0]}</strong> and <strong>{formatted_words[1]}</strong>"
        else:
            # Join all but last with commas, then add "and" before the last word
            word_list = ", ".join([f"<strong>{w}</strong>" for w in formatted_words[:-1]]) + f", and <strong>{formatted_words[-1]}</strong>"

        improvements.append(f"To improve clarity and reduce ambiguity, focus on refining the pronunciation of: {word_list}.")

    return improvements

def generate_corrected_text(transcribed_text):
    """
    Genera la versión gramaticalmente corregida del texto transcrito 
    utilizando un modelo de lenguaje grande (LLM) para corrección avanzada.
    """
    
    # Si la clave API no está configurada, se devuelve el texto sin corregir como fallback
    if not os.getenv("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY not found. Returning uncorrected text.")
        return transcribed_text

    try:
        # Inicializa el cliente (obtiene la clave API automáticamente del entorno)
        client = Client()
        
        # Instrucción de sistema para asegurar que el modelo solo corrija la gramática en español
        system_instruction = (
            "Eres un corrector gramatical experto en español. "
            "Corrige el texto de entrada para mejorar la gramática, ortografía y fluidez. "
            "Devuelve *solamente* el texto corregido sin añadir explicaciones, títulos, ni notas."
        )
        
        # Llama al modelo para realizar la corrección
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Modelo rápido y eficiente para corrección
            contents=[types.Part.from_text(transcribed_text)],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0 # Temperatura baja para resultados deterministas y de alta fidelidad
            )
        )
        
        # Extrae y devuelve el texto corregido
        corrected = response.text.strip()
        
        # Si la respuesta del modelo es vacía, devuelve el texto original
        if not corrected:
            return transcribed_text
            
        return corrected

    except Exception as e:
        # Manejo de errores de la API (p.ej., límite de tokens, error de conexión)
        logger.error(f"Error during LLM correction: {str(e)}")
        return transcribed_text

def generate_tts_feedback(text, score):
    """Generate Text-to-Speech audio feedback in Spanish

    Args:
        text: Text to convert to speech
        score: Assessment score (0-100) to determine speaking rate

    Returns:
        URL to TTS audio file
    """
    try:
        # Initialize Text-to-Speech client
        client = texttospeech.TextToSpeechClient()

        # Select voice based on score (slower for beginners)
        if score < 60:
            speaking_rate = 0.8  # Slow for beginners
        elif score < 75:
            speaking_rate = 0.9  # Moderate for developing speakers
        else:
            speaking_rate = 1.0  # Normal for proficient speakers
        
        # Build the voice request
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Use a female Spanish voice
        voice = texttospeech.VoiceSelectionParams(
            language_code="es-ES",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        # Select the type of audio file
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate
        )
        
        # Perform the text-to-speech request
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        # Generate a unique filename
        filename = f"tts_{uuid.uuid4()}.mp3"
        
        # If we have a bucket, upload to Cloud Storage
        if bucket:
            try:
                blob = bucket.blob(f"tts/{filename}")
                blob.upload_from_bytes(response.audio_content, content_type='audio/mpeg')
                
                # Create a signed URL that will be valid for 2 hours
                url = blob.generate_signed_url(
                    version="v4",
                    expiration=datetime.timedelta(hours=2),
                    method="GET"
                )
                logger.info(f"TTS audio generated and uploaded: {filename}")
                return url
            except Exception as e:
                logger.error(f"Error uploading TTS audio to bucket: {str(e)}")
                # Fallback to local storage if bucket upload fails
                # Cleanup old files periodically
                cleanup_old_tts_files()
                local_filename = f"tts_{uuid.uuid4()}.mp3"
                local_filepath = os.path.join(TTS_TEMP_DIR, local_filename)
                with open(local_filepath, 'wb') as f:
                    f.write(response.audio_content)
                logger.info(f"TTS audio saved locally: {local_filepath}")
                return url_for('get_tts_audio', filename=local_filename)
        else:
            # Save to TTS temp directory
            # Cleanup old files periodically
            cleanup_old_tts_files()
            local_filename = f"tts_{uuid.uuid4()}.mp3"
            local_filepath = os.path.join(TTS_TEMP_DIR, local_filename)
            with open(local_filepath, 'wb') as f:
                f.write(response.audio_content)
            logger.info(f"TTS audio saved locally: {local_filepath}")
            return url_for('get_tts_audio', filename=local_filename)

    except Exception as e:
        logger.error(f"Error generating TTS audio: {str(e)}")
        return None
        
# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    bucket_status = "connected" if bucket else "not connected"
    return jsonify({
        "status": "ok",
        "bucket": bucket_status,
        "bucket_name": BUCKET_NAME,
        "dictionary_size": len(SPANISH_DICT)
    })

@app.route('/process-audio', methods=['POST'])
def process_audio():
    """Process uploaded or recorded audio and provide assessment"""
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({"error": "No audio file in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename in request")
            return jsonify({"error": "No selected file"}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({"error": "Invalid file type. Please upload .wav, .mp3, .m4a, .opus, .webm, or .ogg"}), 400
        
        # Check if this is a practice mode assessment
        practice_level = request.form.get('practice_level', None)

        # Get the user's selected proficiency level (beginner/intermediate/advanced)
        user_level = request.form.get('level', 'intermediate')

        # Validate level parameter
        if user_level not in ['beginner', 'intermediate', 'advanced']:
            user_level = 'intermediate'

        # Extract tracking parameters
        tracking_source = request.form.get('source', 'direct')
        tracking_cohort = request.form.get('cohort', 'none')

        # Process in memory
        audio_content = file.read()
        logger.info(f"Received audio file of size: {len(audio_content)} bytes")

        # Transcribe audio (returns dict with 'transcript' and 'words')
        transcription_data = transcribe_audio(audio_content)
        spoken_text = transcription_data.get('transcript', '')

        if not spoken_text:
            logger.warning("No transcription returned")
            return jsonify({
            "score": 70,
            "transcribed_text": "No se pudo transcribir el audio. Por favor, intente de nuevo con una grabación un poco más corta.",
            "corrected_text": "No transcription available. Please try again with a slightly shorter recording.",
            "error": "Sorry — we're experiencing a temporary technical issue.\nPlease try again with a slightly shorter recording.",
            "feedback": "Our system had difficulty processing your recording. This could be due to a temporary technical issue or the recording being too long.",
            "strengths": ["Attempt to speak in Spanish"],
        "areas_for_improvement": [
            "Try recording for 60-90 seconds or less",
            "Ensure stable internet connection",
            "Use a good quality microphone",
            "Reduce background noise"
            ],
            "tts_audio_url": None
        })

        # Calculate assessment based on mode
        if practice_level and practice_level in REFERENCES:
            # Practice mode with reference phrase
            assessment = assess_practice_phrase(transcription_data, practice_level, level=user_level)
            corrected_text = REFERENCES[practice_level]  # Use reference as corrected text
            logger.info(f"Practice mode assessment: level={user_level}, practice_level={practice_level}, score={assessment['score']}")
        else:
            # Free speech mode
            assessment = assess_free_speech(transcription_data, level=user_level)
            corrected_text = generate_corrected_text(spoken_text)
            logger.info(f"Free speech assessment: level={user_level}, score={assessment['score']}")

        # Generate TTS feedback (pass score for determining speaking rate)
        tts_url = generate_tts_feedback(corrected_text, assessment['score'])

        # Prepare response (NO 'level' shown to user, only score and feedback)
        response = {
            "score": assessment['score'],
            "transcribed_text": spoken_text,
            "corrected_text": corrected_text,
            "feedback": assessment['feedback'],
            "strengths": assessment['strengths'],
            "areas_for_improvement": assessment['areas_for_improvement'],
            "tts_audio_url": tts_url
        }

        # Add FACT breakdown if available (for transparency)
        if 'fact_breakdown' in assessment:
            response["fact_breakdown"] = assessment['fact_breakdown']

        # Add practice-specific fields if applicable
        if practice_level and 'reference_text' in assessment:
            response["reference_text"] = assessment['reference_text']
            response["reference_similarity"] = assessment.get('reference_similarity')

        # Send tracking data to Google Sheets webhook (non-blocking)
        if TRACKING_WEBHOOK_URL:
            try:
                # Calculate recording duration from transcription timing
                words_data = transcription_data.get('words', [])
                duration_seconds = 0
                if words_data and len(words_data) > 0:
                    duration_seconds = round(words_data[-1]['end_time'] - words_data[0]['start_time'], 1)

                # Prepare tracking data
                tracking_data = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'source': tracking_source,
                    'cohort': tracking_cohort,
                    'duration_seconds': duration_seconds,
                    'score': round(assessment['score'], 2)
                }

                # Send to webhook (with short timeout to avoid blocking user response)
                requests.post(TRACKING_WEBHOOK_URL, json=tracking_data, timeout=3)
                logger.info(f"Tracking data sent: source={tracking_source}, cohort={tracking_cohort}, duration={duration_seconds}s, score={assessment['score']}")
            except Exception as e:
                # Log error but don't fail the request if tracking fails
                logger.error(f"Failed to send tracking data to webhook: {str(e)}")

        return jsonify(response)
            
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-tts-audio/<filename>')
def get_tts_audio(filename):
    """Serve TTS audio files from local TTS temp directory"""
    # Security: Validate filename to prevent path traversal attacks
    if '/' in filename or '\\' in filename or '..' in filename:
        logger.warning(f"Invalid TTS filename requested: {filename}")
        return "Invalid filename", 400

    # Only allow expected filename pattern (tts_<uuid>.mp3)
    if not filename.startswith('tts_') or not filename.endswith('.mp3'):
        logger.warning(f"Unexpected TTS filename format: {filename}")
        return "Invalid filename format", 400

    file_path = os.path.join(TTS_TEMP_DIR, filename)

    # Verify the file exists and is within the TTS directory
    if not os.path.exists(file_path):
        return "Audio file not found", 404

    # Extra security: ensure resolved path is within TTS_TEMP_DIR
    real_path = os.path.realpath(file_path)
    real_tts_dir = os.path.realpath(TTS_TEMP_DIR)
    if not real_path.startswith(real_tts_dir):
        logger.warning(f"Path traversal attempt detected: {filename}")
        return "Invalid path", 400

    return send_file(file_path, mimetype='audio/mpeg')

@app.route('/references')
def get_references():
    """Serves the reference phrases for practice"""
    try:
        return jsonify(REFERENCES)
    except Exception as e:
        logger.error(f"Error loading references: {e}")
        return jsonify({
            "error": "Could not load references",
            "short": "Hola, ¿cómo estás?"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

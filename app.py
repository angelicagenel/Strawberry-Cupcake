import os
from google.genai import types
from google.genai import Client
import json
import tempfile
import logging
import datetime
import re
import statistics
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

# Maximum file size (20MB)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'opus', 'webm', 'ogg'}

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

def evaluate_pronunciation_fluency(transcript, words_data):
    """C1: Pronunciation Behavior (30% weight)

    Evaluates functional clarity through 4 subcriteria:
    1. Rhythm Stability - WPS standard deviation
    2. Flow Continuity - Internal vs strategic pauses
    3. Vowel Duration - WPS drops without pauses (vowel dragging)
    4. Global Stability - STT confidence (acts as ceiling/floor, NOT direct penalty)

    Args:
        transcript: Full transcribed text
        words_data: List of word objects with timing and confidence

    Returns:
        dict with 'score' (0-100), 'details', and 'patterns_activated'
    """
    if not words_data or len(words_data) == 0:
        return {
            'score': 70,
            'details': {
                'rhythm_stability': 70,
                'flow_continuity': 70,
                'vowel_duration': 70,
                'global_stability': 70,
                'note': 'No timing data available'
            },
            'patterns_activated': []
        }

    patterns_activated = []

    # --- SUBCRITERION 1.1: RHYTHM STABILITY ---
    # Measure WPS standard deviation to detect erratic rhythm
    try:
        # Calculate WPS in 3-second windows
        duration = words_data[-1]['end_time'] - words_data[0]['start_time']
        if duration > 3:
            window_wps = []
            window_size = 3.0  # 3-second windows
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

                # Score based on deviation
                if wps_std_dev < 0.3:
                    rhythm_score = 95
                elif wps_std_dev < 0.6:
                    rhythm_score = 75
                    patterns_activated.append(('even_syllable_rhythm', 'medium'))
                else:
                    rhythm_score = 55
                    patterns_activated.append(('even_syllable_rhythm', 'high'))
            else:
                rhythm_score = 75
                wps_std_dev = 0
        else:
            # Too short for window analysis
            rhythm_score = 75
            wps_std_dev = 0
    except:
        rhythm_score = 70
        wps_std_dev = 0

    # --- SUBCRITERION 1.2: FLOW CONTINUITY ---
    # Distinguish internal pauses (bad) from strategic pauses (good)
    try:
        internal_pauses = 0
        strategic_pauses = 0

        for i in range(len(words_data) - 1):
            gap = words_data[i+1]['start_time'] - words_data[i]['end_time']

            if gap > 1.5:  # Long pause detected
                # Heuristic: pause after connector/end-of-idea words = strategic
                current_word = words_data[i]['word'].lower()
                strategic_markers = ['.', 'entonces', 'luego', 'finalmente', 'además', 'pero']

                is_strategic = any(marker in current_word for marker in strategic_markers)

                if is_strategic:
                    strategic_pauses += 1
                else:
                    internal_pauses += 1

        # Score based on internal vs strategic
        if internal_pauses == 0:
            flow_score = 95
        elif internal_pauses <= 2:
            flow_score = 75
            patterns_activated.append(('natural_pausing', 'medium'))
        else:
            flow_score = 55
            patterns_activated.append(('natural_pausing', 'high'))

    except:
        flow_score = 70
        internal_pauses = 0
        strategic_pauses = 0

    # --- SUBCRITERION 1.3: VOWEL DURATION ---
    # Detect WPS drops without corresponding long pauses (vowel dragging)
    try:
        wps_collapses = 0

        if len(words_data) >= 4:
            # Calculate WPS for each word pair
            for i in range(len(words_data) - 3):
                # Get WPS for segments of 2 words each
                segment1_duration = words_data[i+1]['end_time'] - words_data[i]['start_time']
                segment2_duration = words_data[i+3]['end_time'] - words_data[i+2]['start_time']

                if segment1_duration > 0 and segment2_duration > 0:
                    wps1 = 2 / segment1_duration
                    wps2 = 2 / segment2_duration

                    # Detect collapse (>40% drop)
                    if wps2 < wps1 * 0.6:
                        # Check if there's a long pause explaining the drop
                        gap = words_data[i+2]['start_time'] - words_data[i+1]['end_time']
                        if gap < 1.5:  # No long pause = vowel dragging
                            wps_collapses += 1

        if wps_collapses == 0:
            vowel_score = 95
        elif wps_collapses <= 2:
            vowel_score = 75
            patterns_activated.append(('pure_vowels', 'medium'))
        else:
            vowel_score = 55
            patterns_activated.append(('pure_vowels', 'high'))

    except:
        vowel_score = 70
        wps_collapses = 0

    # --- SUBCRITERION 1.4: GLOBAL STABILITY (STT Confidence) ---
    # This is used as a CEILING/FLOOR, not a direct score
    try:
        confidences = [w['confidence'] for w in words_data if 'confidence' in w]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            stt_confidence_pct = avg_confidence * 100
        else:
            stt_confidence_pct = 75
    except:
        stt_confidence_pct = 75

    # --- CALCULATE C1 SCORE ---
    # Average of 3 behavioral subcriteria (NOT including STT confidence directly)
    base_c1_score = (rhythm_score + flow_score + vowel_score) / 3

    # --- APPLY STT CONFIDENCE AS CEILING ---
    # STT confidence determines maximum possible score
    if stt_confidence_pct >= 80:
        # No restriction
        c1_score = base_c1_score
    elif stt_confidence_pct >= 70:
        # Cap at 85
        c1_score = min(85, base_c1_score)
        patterns_activated.append(('fundamentals_needed', 'medium'))
    else:
        # Cap at 70, activate fundamentals
        c1_score = min(70, base_c1_score)
        patterns_activated.append(('fundamentals_needed', 'high'))

    return {
        'score': round(c1_score, 1),
        'subcriteria': {
            'c1_1_rhythm_stability': round(rhythm_score, 1),
            'c1_2_flow_continuity': round(flow_score, 1),
            'c1_3_vowel_duration': round(vowel_score, 1),
            'c1_4_global_stability': round(stt_confidence_pct, 1)
        },
        'details': {
            'rhythm_stability': round(rhythm_score, 1),
            'flow_continuity': round(flow_score, 1),
            'vowel_duration': round(vowel_score, 1),
            'global_stability': round(stt_confidence_pct, 1),
            'wps_std_dev': round(wps_std_dev, 2) if wps_std_dev else 0,
            'internal_pauses': internal_pauses,
            'strategic_pauses': strategic_pauses,
            'wps_collapses': wps_collapses
        },
        'patterns_activated': patterns_activated
    }


def evaluate_functions(transcript, level='intermediate'):
    """C2: Functional Language Control (25% weight)

    Evaluates communicative intent through 4 EXPLICIT subcriteria:
    - C2.1 Level Adequacy - Uses expected functions for the level
    - C2.2 Functional Consistency - Function appears ≥3 times coherently
    - C2.3 Dominant Function - Clear whether describe/narrate/opine/justify
    - C2.4 Modality (3 layers) - Basic/cognitive/evaluative

    Key principle: Subjunctive is a BONUS signal, not a requirement.
    Evaluative modality without perfect subjunctive still counts.

    Args:
        transcript: Full transcribed text
        level: Expected level (beginner/intermediate/advanced)

    Returns:
        dict with 'score' (0-100), 'subcriteria', 'detected', 'modality_detected'
    """
    text_lower = transcript.lower()
    detected = {}
    modality_detected = {}

    # --- DETECT BASIC STRUCTURES ---
    # Present (always relevant)
    present_patterns = r'\b(soy|estoy|tengo|hablo|vivo|trabajo|estudio|como|hago|voy|es|son)\b'
    present_matches = re.findall(present_patterns, text_lower)
    detected['present'] = len(present_matches)

    # Past tense (preterite)
    preterite_patterns = r'\b(fui|hice|comí|dije|fue|hablé|estudié|trabajé|viví|tuve|estuvo|hizo|desperté|regresé)\b'
    preterite_matches = re.findall(preterite_patterns, text_lower)
    detected['preterite'] = len(preterite_matches)

    # Past tense (imperfect)
    imperfect_patterns = r'\b(era|estaba|tenía|iba|hacía|hablaba|comía|vivía|trabajaba|estudiaba)\b'
    imperfect_matches = re.findall(imperfect_patterns, text_lower)
    detected['imperfect'] = len(imperfect_matches)

    # Future
    future_patterns = r'\b(voy a|va a|vamos a|iré|será|haré|tendré|estaré|podré)\b'
    future_matches = re.findall(future_patterns, text_lower)
    detected['future'] = len(future_matches)

    # Subjunctive (BONUS signal, not requirement)
    subjunctive_patterns = r'\b(sea|esté|tenga|quiera|pueda|haya|espero que|es importante que|me preocupa que|no creo que|ojalá|para que)\b'
    subjunctive_matches = re.findall(subjunctive_patterns, text_lower)
    detected['subjunctive'] = len(subjunctive_matches)

    # Conditional
    conditional_patterns = r'\b(sería|haría|iría|tendría|podría|debería|si fuera|si tuviera|si pudiera)\b'
    conditional_matches = re.findall(conditional_patterns, text_lower)
    detected['conditional'] = len(conditional_matches)

    # --- DETECT MODALITY (3 LAYERS) ---
    # Layer 1: Basic modality (preferences, abilities)
    basic_modality = re.findall(MODALITY_LAYERS['basic']['patterns'], text_lower)
    modality_detected['basic'] = len(basic_modality)

    # Layer 2: Cognitive modality (opinions, beliefs)
    cognitive_modality = re.findall(MODALITY_LAYERS['cognitive']['patterns'], text_lower)
    modality_detected['cognitive'] = len(cognitive_modality)

    # Layer 3: Evaluative modality (judgments, norms)
    evaluative_modality = re.findall(MODALITY_LAYERS['evaluative']['patterns'], text_lower)
    modality_detected['evaluative'] = len(evaluative_modality)

    # ========================================================================
    # SUBCRITERION C2.1: LEVEL ADEQUACY
    # Does the speaker use functions expected for their level?
    # ========================================================================
    c2_1_level_adequacy = 50  # Base

    if level == 'beginner':
        # Beginner: Present tense expected
        if detected['present'] >= 5:
            c2_1_level_adequacy = 95
        elif detected['present'] >= 3:
            c2_1_level_adequacy = 75
        elif detected['present'] >= 1:
            c2_1_level_adequacy = 60
        else:
            c2_1_level_adequacy = 50

    elif level == 'intermediate':
        # Intermediate: Past tense (preterite or imperfect) expected
        past_total = detected['preterite'] + detected['imperfect']
        if past_total >= 5:
            c2_1_level_adequacy = 95
        elif past_total >= 3:
            c2_1_level_adequacy = 75
        elif past_total >= 1:
            c2_1_level_adequacy = 60
        else:
            c2_1_level_adequacy = 50

    elif level == 'advanced':
        # Advanced: Complex structures (subjunctive, conditional, or cognitive modality)
        complex_total = detected['subjunctive'] + detected['conditional']
        has_cognitive_or_evaluative = modality_detected['cognitive'] >= 1 or modality_detected['evaluative'] >= 1

        if complex_total >= 2 and has_cognitive_or_evaluative:
            c2_1_level_adequacy = 95
        elif complex_total >= 1 or has_cognitive_or_evaluative:
            c2_1_level_adequacy = 75
        else:
            c2_1_level_adequacy = 60

    # ========================================================================
    # SUBCRITERION C2.2: FUNCTIONAL CONSISTENCY
    # Is the same function sustained ≥3 times?
    # ========================================================================
    c2_2_functional_consistency = 50  # Base

    # Count sustained use of any single function
    max_single_function = max([
        detected['present'],
        detected['preterite'],
        detected['imperfect'],
        detected['future'],
        detected['subjunctive'],
        detected['conditional']
    ])

    if max_single_function >= 5:
        c2_2_functional_consistency = 95
    elif max_single_function >= 3:
        c2_2_functional_consistency = 75
    elif max_single_function >= 2:
        c2_2_functional_consistency = 60
    else:
        c2_2_functional_consistency = 50

    # ========================================================================
    # SUBCRITERION C2.3: DOMINANT FUNCTION
    # Is it clear if speaker is describing / narrating / opining / justifying?
    # ========================================================================
    c2_3_dominant_function = 50  # Base

    # Determine dominant function based on detected structures
    function_clarity = 'unclear'

    if detected['present'] > detected['preterite'] + detected['imperfect'] and detected['present'] >= 3:
        function_clarity = 'describe'  # Describing current state
    elif (detected['preterite'] + detected['imperfect']) >= 3:
        function_clarity = 'narrate'  # Narrating past events
    elif modality_detected['cognitive'] >= 2 or modality_detected['evaluative'] >= 1:
        function_clarity = 'opine'  # Expressing opinions/evaluations
    elif detected['subjunctive'] >= 2 or detected['conditional'] >= 2:
        function_clarity = 'justify'  # Justifying with complex modality

    if function_clarity in ['describe', 'narrate', 'opine', 'justify']:
        c2_3_dominant_function = 90  # Clear dominant function
    elif function_clarity == 'unclear' and max_single_function >= 2:
        c2_3_dominant_function = 70  # Some clarity but diffuse
    else:
        c2_3_dominant_function = 55  # Confused

    # ========================================================================
    # SUBCRITERION C2.4: MODALITY (3 LAYERS)
    # Basic (0.3) / Cognitive (0.6) / Evaluative (1.0)
    # ========================================================================
    c2_4_modality = 50  # Base

    modality_points = 0

    if modality_detected['basic'] >= 1:
        modality_points += 15  # Basic modality present
    if modality_detected['cognitive'] >= 1:
        modality_points += 20  # Cognitive modality present
    if modality_detected['evaluative'] >= 1:
        modality_points += 25  # Evaluative modality present (highest)

    c2_4_modality = min(100, 50 + modality_points)

    # ========================================================================
    # CALCULATE C2 FINAL SCORE (average of 4 subcriteria)
    # ========================================================================
    c2_final_score = (c2_1_level_adequacy + c2_2_functional_consistency +
                      c2_3_dominant_function + c2_4_modality) / 4

    return {
        'score': round(c2_final_score, 1),
        'subcriteria': {
            'c2_1_level_adequacy': round(c2_1_level_adequacy, 1),
            'c2_2_functional_consistency': round(c2_2_functional_consistency, 1),
            'c2_3_dominant_function': round(c2_3_dominant_function, 1),
            'c2_4_modality': round(c2_4_modality, 1)
        },
        'detected': detected,
        'modality_detected': modality_detected,
        'dominant_function': function_clarity
    }


def evaluate_text_type(transcript, words_data=None):
    """C3: Discourse Organization (20% weight)

    Evaluates prosodic discourse structure through 4 EXPLICIT subcriteria:
    - C3.1 Sequence - Temporal connectors OR strategic pauses
    - C3.2 Cohesion - Functional connectors (causal, adversative, additive)
    - C3.3 Development - Idea length + elaboration
    - C3.4 Text Type - Matches prompt intent

    Key principle: A "functional sentence" can exist without a pause if there's a connector.
    Long fluent discourse is NOT penalized if it maintains semantic coherence.

    Args:
        transcript: Full transcribed text
        words_data: Word timing data (optional, for pause-based analysis)

    Returns:
        dict with 'score' (0-100), 'subcriteria', 'details'
    """
    words = transcript.split()
    word_count = len(words)
    text_lower = transcript.lower()

    # --- DETECT CONNECTORS (by type) ---
    connector_counts = {}
    total_connectors = 0

    for conn_type, conn_list in CONNECTOR_TYPES.items():
        count = sum(1 for c in conn_list if c in text_lower)
        connector_counts[conn_type] = count
        total_connectors += count

    # Connector variety (how many types used)
    connector_variety = sum(1 for count in connector_counts.values() if count > 0)

    # --- COUNT FUNCTIONAL SENTENCES ---
    # Method 1: Use strategic pauses if available
    if words_data and len(words_data) > 0:
        functional_sentences = 1  # Start with 1 (first utterance)
        strategic_pauses = 0

        for i in range(len(words_data) - 1):
            gap = words_data[i+1]['start_time'] - words_data[i]['end_time']

            # Strategic pause = end of sentence
            if gap > 1.5:
                current_word = words_data[i]['word'].lower()
                # Check if pause follows a connector or end-of-idea marker
                strategic_markers = ['entonces', 'luego', 'finalmente', 'además', 'pero', 'porque']
                if any(marker in current_word for marker in strategic_markers):
                    functional_sentences += 1
                    strategic_pauses += 1

    # Method 2: Fallback - count by connectors
    else:
        # Each connector implies a sentence boundary
        functional_sentences = 1 + (total_connectors if total_connectors > 0 else 0)
        strategic_pauses = 0

    # --- DETECT DISCOURSE TYPE ---
    discourse_type = 'conversational'

    # Academic discourse markers
    academic_markers = ['considero que', 'es importante', 'es necesario', 'me parece importante',
                        'por lo tanto', 'sin embargo', 'no obstante', 'debido a']
    academic_count = sum(1 for marker in academic_markers if marker in text_lower)

    # Narrative markers
    narrative_markers = ['ayer', 'primero', 'después', 'luego', 'entonces', 'finalmente']
    narrative_count = sum(1 for marker in narrative_markers if marker in text_lower)

    # Argumentative markers
    argumentative_markers = ['creo que', 'pienso que', 'me preocupa', 'aunque', 'sin embargo', 'por un lado']
    argumentative_count = sum(1 for marker in argumentative_markers if marker in text_lower)

    # Determine dominant type
    if academic_count >= 2:
        discourse_type = 'academic'
    elif argumentative_count >= 2:
        discourse_type = 'argumentative'
    elif narrative_count >= 3:
        discourse_type = 'narrative'
    else:
        discourse_type = 'descriptive'

    # ========================================================================
    # SUBCRITERION C3.1: SEQUENCE
    # Evidence of temporal/logical order via connectors OR strategic pauses
    # ========================================================================
    c3_1_sequence = 50  # Base

    temporal_connectors = connector_counts.get('temporal', 0)

    if temporal_connectors >= 3 or (strategic_pauses >= 2 and temporal_connectors >= 1):
        c3_1_sequence = 95  # Clear sequence
    elif temporal_connectors >= 2 or strategic_pauses >= 2:
        c3_1_sequence = 75  # Partial sequence
    elif temporal_connectors >= 1 or total_connectors >= 2:
        c3_1_sequence = 65  # Some sequence
    else:
        c3_1_sequence = 50  # No sequence

    # ========================================================================
    # SUBCRITERION C3.2: COHESION
    # Functional use of connectors (causal, adversative, additive)
    # ========================================================================
    c3_2_cohesion = 50  # Base

    functional_connector_count = (connector_counts.get('causal', 0) +
                                  connector_counts.get('adversative', 0) +
                                  connector_counts.get('additive', 0))

    if functional_connector_count >= 3 and connector_variety >= 2:
        c3_2_cohesion = 95  # Functional cohesion
    elif functional_connector_count >= 2:
        c3_2_cohesion = 75  # Some cohesion
    elif functional_connector_count >= 1 or total_connectors >= 2:
        c3_2_cohesion = 65  # Mechanical cohesion
    else:
        c3_2_cohesion = 50  # Absent

    # ========================================================================
    # SUBCRITERION C3.3: DEVELOPMENT
    # Ideas are elaborated (not just lists)
    # ========================================================================
    c3_3_development = 50  # Base

    # Heuristic: word_count / functional_sentences = average idea length
    if functional_sentences > 0:
        avg_idea_length = word_count / functional_sentences
    else:
        avg_idea_length = word_count

    if avg_idea_length >= 15 and functional_sentences >= 3:
        c3_3_development = 95  # Developed ideas
    elif avg_idea_length >= 10 and functional_sentences >= 2:
        c3_3_development = 75  # Some development
    elif avg_idea_length >= 5:
        c3_3_development = 65  # Limited development
    else:
        c3_3_development = 50  # Fragmented (lists)

    # ========================================================================
    # SUBCRITERION C3.4: TEXT TYPE
    # Matches expected discourse type for prompt
    # ========================================================================
    c3_4_text_type = 50  # Base

    # Appropriate discourse type = high score
    if discourse_type in ['narrative', 'argumentative', 'academic']:
        c3_4_text_type = 90  # Appropriate type
    elif discourse_type == 'descriptive' and word_count >= 30:
        c3_4_text_type = 75  # Descriptive but extended
    else:
        c3_4_text_type = 65  # Conversational/simple

    # ========================================================================
    # CALCULATE C3 FINAL SCORE (average of 4 subcriteria)
    # ========================================================================
    c3_final_score = (c3_1_sequence + c3_2_cohesion +
                      c3_3_development + c3_4_text_type) / 4

    return {
        'score': round(c3_final_score, 1),
        'subcriteria': {
            'c3_1_sequence': round(c3_1_sequence, 1),
            'c3_2_cohesion': round(c3_2_cohesion, 1),
            'c3_3_development': round(c3_3_development, 1),
            'c3_4_text_type': round(c3_4_text_type, 1)
        },
        'details': {
            'word_count': word_count,
            'functional_sentences': functional_sentences,
            'total_connectors': total_connectors,
            'connector_variety': connector_variety,
            'connector_counts': connector_counts,
            'discourse_type': discourse_type
        }
    }


def evaluate_context(transcript, level='intermediate'):
    """C4: Contextual Lexical Richness (15% weight)

    Evaluates integrated vocabulary use through 3 EXPLICIT subcriteria:
    - C4.1 Variety Ratio - Unique content words / total content words (pure metric)
    - C4.2 Thematic Alignment - Does vocabulary serve the topic?
    - C4.3 Relative Progression - Personal/everyday/abstract appropriate to level

    Key principle: Progression is relative to level.
    - Beginner: Personal vocabulary = high score
    - Intermediate: Everyday vocabulary = high score
    - Advanced: Abstract vocabulary = high score

    Args:
        transcript: Full transcribed text
        level: Expected level (beginner/intermediate/advanced)

    Returns:
        dict with 'score' (0-100), 'subcriteria', 'details', 'thematic_level'
    """
    words = transcript.lower().split()
    if not words:
        return {
            'score': 50,
            'subcriteria': {
                'c4_1_variety_ratio': 50,
                'c4_2_thematic_alignment': 50,
                'c4_3_relative_progression': 50
            },
            'details': {},
            'thematic_level': 'none'
        }

    # --- CALCULATE VARIETY RATIO ---
    # Remove punctuation from words
    clean_words = [re.sub(r'[^\w\s]', '', w) for w in words]
    clean_words = [w for w in clean_words if w]

    # Exclude common function words from variety calculation
    function_words = ['el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'a', 'al',
                     'en', 'con', 'por', 'para', 'que', 'y', 'o', 'pero', 'es', 'son']
    content_words = [w for w in clean_words if w not in function_words]

    if len(content_words) == 0:
        variety_ratio = 0.5
    else:
        unique_content = set(content_words)
        variety_ratio = len(unique_content) / len(content_words)

    # --- DETECT THEMATIC LEVEL ---
    text_lower = transcript.lower()

    # Personal vocabulary
    personal_words = ['yo', 'mi', 'me', 'mis', 'nombre', 'soy', 'tengo', 'familia', 'años', 'casa', 'amigo']
    personal_count = sum(1 for w in personal_words if w in text_lower)

    # Everyday vocabulary
    everyday_words = ['trabajo', 'escuela', 'comer', 'estudiar', 'tiempo', 'día', 'hacer', 'ir',
                     'desperté', 'desayuné', 'almorcé', 'regresé', 'hablé']
    everyday_count = sum(1 for w in everyday_words if w in text_lower)

    # Abstract/academic vocabulary (BONUS signal)
    abstract_words = ['sociedad', 'cultura', 'problema', 'educación', 'importante', 'necesario',
                     'tecnología', 'futuro', 'desarrollar', 'híbrido', 'considero', 'democratiza',
                     'acentúa', 'brecha', 'integrando']
    abstract_count = sum(1 for w in abstract_words if w in text_lower)

    # Determine thematic level
    if abstract_count >= 2:
        thematic_level = 'abstract'
    elif everyday_count >= 3:
        thematic_level = 'everyday'
    elif personal_count >= 2:
        thematic_level = 'personal'
    else:
        thematic_level = 'basic'

    # ========================================================================
    # SUBCRITERION C4.1: VARIETY RATIO (Pure Metric)
    # Unique content words / total content words
    # ========================================================================
    c4_1_variety_ratio = 50  # Base

    if variety_ratio >= 0.70:
        c4_1_variety_ratio = 95  # High variety
    elif variety_ratio >= 0.60:
        c4_1_variety_ratio = 75  # Good variety
    elif variety_ratio >= 0.50:
        c4_1_variety_ratio = 65  # Moderate variety
    else:
        c4_1_variety_ratio = 50  # Low variety

    # ========================================================================
    # SUBCRITERION C4.2: THEMATIC ALIGNMENT
    # Does the vocabulary serve the topic? (Integrated vs off-topic)
    # ========================================================================
    c4_2_thematic_alignment = 50  # Base

    # Check if vocabulary is coherent with any theme
    total_thematic_words = personal_count + everyday_count + abstract_count

    if total_thematic_words >= 5 and thematic_level != 'basic':
        c4_2_thematic_alignment = 95  # Integrated vocabulary
    elif total_thematic_words >= 3:
        c4_2_thematic_alignment = 75  # Partial integration
    elif total_thematic_words >= 1:
        c4_2_thematic_alignment = 65  # Some thematic words
    else:
        c4_2_thematic_alignment = 50  # Off-topic or generic

    # ========================================================================
    # SUBCRITERION C4.3: RELATIVE PROGRESSION
    # Is vocabulary level-appropriate?
    # Personal (beginner) / Everyday (intermediate) / Abstract (advanced)
    # ========================================================================
    c4_3_relative_progression = 50  # Base

    if level == 'beginner':
        # Beginner: Personal vocabulary = high score
        if thematic_level == 'personal':
            c4_3_relative_progression = 95  # Perfect match
        elif thematic_level in ['everyday', 'abstract']:
            c4_3_relative_progression = 80  # Going beyond (bonus)
        else:
            c4_3_relative_progression = 60  # Basic

    elif level == 'intermediate':
        # Intermediate: Everyday vocabulary = high score
        if thematic_level == 'everyday':
            c4_3_relative_progression = 95  # Perfect match
        elif thematic_level == 'abstract':
            c4_3_relative_progression = 90  # Going beyond (bonus)
        elif thematic_level == 'personal':
            c4_3_relative_progression = 70  # Below expected (not penalized heavily)
        else:
            c4_3_relative_progression = 60  # Basic

    elif level == 'advanced':
        # Advanced: Abstract vocabulary = high score
        if thematic_level == 'abstract':
            c4_3_relative_progression = 95  # Perfect match
        elif thematic_level == 'everyday':
            c4_3_relative_progression = 70  # Below expected (not ideal)
        elif thematic_level == 'personal':
            c4_3_relative_progression = 60  # Well below expected
        else:
            c4_3_relative_progression = 55  # Basic

    # ========================================================================
    # CALCULATE C4 FINAL SCORE (average of 3 subcriteria)
    # ========================================================================
    c4_final_score = (c4_1_variety_ratio + c4_2_thematic_alignment +
                      c4_3_relative_progression) / 3

    return {
        'score': round(c4_final_score, 1),
        'subcriteria': {
            'c4_1_variety_ratio': round(c4_1_variety_ratio, 1),
            'c4_2_thematic_alignment': round(c4_2_thematic_alignment, 1),
            'c4_3_relative_progression': round(c4_3_relative_progression, 1)
        },
        'details': {
            'variety_ratio': round(variety_ratio, 2),
            'unique_content_words': len(set(content_words)),
            'total_content_words': len(content_words),
            'thematic_counts': {
                'personal': personal_count,
                'everyday': everyday_count,
                'abstract': abstract_count
            }
        },
        'thematic_level': thematic_level
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
    """Main FACT assessment function based on instructor's rubric

    NEW Weights (final specification):
    - C1: Pronunciation Behavior: 30%
    - C2: Functional Language Control: 25%
    - C3: Discourse Organization: 20%
    - C4: Contextual Lexical Richness: 15%
    - C5: Prompt Alignment: 10%

    Score ranges:
    - 85-100: Exceeds Expectations
    - 75-84: Meets Expectations
    - 60-74: Partially Meets Expectations
    - 0-59: Does Not Meet Expectations

    Args:
        transcription_data: dict with 'transcript' and 'words'
        level: Expected level (beginner/intermediate/advanced)
        prompt_type: Type of prompt for C5 alignment check

    Returns:
        dict with score, feedback, strengths, areas_for_improvement, patterns
    """
    transcript = transcription_data.get('transcript', '')
    words_data = transcription_data.get('words', [])

    if not transcript:
        return {
            'score': 70,
            'feedback': "We couldn't detect your speech. Please ensure your microphone is working and try speaking clearly.",
            'strengths': [],
            'areas_for_improvement': ["Check microphone connection and reduce background noise"],
            'fact_breakdown': {},
            'patterns': []
        }

    # --- EVALUATE EACH FACT COMPONENT ---
    c1_pronunciation = evaluate_pronunciation_fluency(transcript, words_data)
    c2_functions = evaluate_functions(transcript, level=level)
    c3_text_type = evaluate_text_type(transcript, words_data=words_data)
    c4_context = evaluate_context(transcript, level=level)
    c5_alignment = evaluate_prompt_alignment(
        transcript,
        c1_pronunciation['score'],
        c2_functions['score'],
        c3_text_type['score'],
        c4_context['score'],
        prompt_type=prompt_type
    )

    # --- CALCULATE WEIGHTED FINAL SCORE ---
    base_final_score = (
        c1_pronunciation['score'] * 0.30 +
        c2_functions['score'] * 0.25 +
        c3_text_type['score'] * 0.20 +
        c4_context['score'] * 0.15 +
        c5_alignment['score'] * 0.10
    )

    # --- APPLY LEVEL MULTIPLIERS (for dynamic thresholds) ---
    # Multipliers only apply to penalties for issues, not to boost scores
    level_multiplier = LEVEL_CONFIGS.get(level, {}).get('level_multiplier', 1.0)

    # Calculate penalty based on distance from 100
    distance_from_perfect = 100 - base_final_score
    adjusted_penalty = distance_from_perfect * level_multiplier
    final_score = max(0, 100 - adjusted_penalty)

    # --- CAPA 3: DIAGNOSTIC LAYER (patterns NEVER affect score) ---
    # Collect all signals from CAPA 2
    raw_signals = c1_pronunciation.get('patterns_activated', [])

    # Call diagnostic engine
    diagnostic_patterns = diagnose_patterns(raw_signals, final_score, level)

    # --- Generate simple feedback (CAPA 4 placeholder) ---
    # Per spec: Only score + max 3 diagnostic labels, NO explanations yet
    feedback_text = _generate_simple_feedback(final_score)
    diagnostic_labels = _generate_diagnostic_labels(diagnostic_patterns)

    logger.info(f"FACT Assessment (Level: {level}) - "
                f"C1: {c1_pronunciation['score']}, C2: {c2_functions['score']}, "
                f"C3: {c3_text_type['score']}, C4: {c4_context['score']}, "
                f"C5: {c5_alignment['score']}, Final: {final_score}, "
                f"Patterns: {len(diagnostic_patterns)}")

    return {
        'score': round(final_score, 1),
        'feedback': feedback_text,
        'strengths': [],  # Empty for now (CAPA 4 placeholder)
        'areas_for_improvement': diagnostic_labels,  # Max 3 diagnostic labels
        'fact_breakdown': {
            'pronunciation_behavior': c1_pronunciation['score'],
            'functional_control': c2_functions['score'],
            'discourse_organization': c3_text_type['score'],
            'lexical_richness': c4_context['score'],
            'prompt_alignment': c5_alignment['score']
        },
        'subcriteria_breakdown': {
            'c1_subcriteria': c1_pronunciation.get('subcriteria', {}),
            'c2_subcriteria': c2_functions.get('subcriteria', {}),
            'c3_subcriteria': c3_text_type.get('subcriteria', {}),
            'c4_subcriteria': c4_context.get('subcriteria', {})
        },
        'diagnostic_patterns': diagnostic_patterns,  # CAPA 3 output
        'details': {
            'c1_details': c1_pronunciation.get('details', {}),
            'c2_detected': c2_functions.get('detected', {}),
            'c2_dominant_function': c2_functions.get('dominant_function', 'unknown'),
            'c3_discourse': c3_text_type.get('details', {}),
            'c4_thematic': c4_context.get('thematic_level', 'unknown'),
            'c5_fulfillment': c5_alignment.get('details', {})
        }
    }


def _generate_simple_feedback(score):
    """CAPA 4 Placeholder - Simple score-based feedback (no ACTFL bands shown to user)

    Per spec: Short feedback based on score, NO level labels visible
    """
    if score >= 85:
        return "Excellent work - you communicate clearly and confidently with strong control of Spanish structures."
    elif score >= 75:
        return "Good work - you communicate effectively with clear pronunciation and good structural control."
    elif score >= 60:
        return "You're making progress - continue practicing to improve fluency, clarity, and grammatical consistency."
    else:
        return "Keep practicing - focus on forming complete sentences, improving pronunciation clarity, and building vocabulary."


def _generate_diagnostic_labels(diagnostic_patterns):
    """CAPA 4 Placeholder - Generate simple diagnostic labels (max 3)

    Per spec: Only diagnostic labels WITHOUT explanations, coaching, or links

    Args:
        diagnostic_patterns: List from CAPA 3 diagnose_patterns()

    Returns:
        List of strings (diagnostic labels only)

    Example output:
        [
            "Rhythm consistency",
            "Natural pausing"
        ]
    """
    if not diagnostic_patterns:
        return ["Continue refining your pronunciation"]

    labels = []
    for pattern in diagnostic_patterns[:3]:  # Max 3
        # Use pattern name only (no diagnostic message, no explanation)
        labels.append(pattern['name'])

    return labels


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
            
            # Check against a sample of the dictionary for performance
            dict_sample = set(list(SPANISH_DICT)[:1000]) if len(SPANISH_DICT) > 1000 else SPANISH_DICT
            
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
    if "Profile C" in level or score >= 85:
        return "Your pronunciation is clear and generally consistent. Small refinements will help improve overall naturalness and ease of understanding."

    # Profile B — Functional Clarity (65-84)
    elif "Profile B" in level or score >= 65:
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
        print("Warning: GEMINI_API_KEY no encontrada. Devolviendo texto sin corregir.")
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
        print(f"Error durante la corrección con el LLM: {e}") 
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
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_file.write(response.audio_content)
                temp_file.close()
                filename = os.path.basename(temp_file.name)
                app.config[f'TTS_FILE_{filename}'] = temp_file.name
                logger.info(f"TTS audio saved locally: {temp_file.name}")
                return url_for('get_tts_audio', filename=filename)
        else:
            # Save to a temporary file and return its path
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file.write(response.audio_content)
            temp_file.close()
            filename = os.path.basename(temp_file.name)
            app.config[f'TTS_FILE_{filename}'] = temp_file.name
            logger.info(f"TTS audio saved locally: {temp_file.name}")
            return url_for('get_tts_audio', filename=filename)

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
        
        return jsonify(response)
            
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-tts-audio/<filename>')
def get_tts_audio(filename):
    """Serve TTS audio files from local storage"""
    file_path = app.config.get(f'TTS_FILE_{filename}')
    if not file_path:
        return "Audio file not found", 404
    
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

import os
from google.genai import types
from google.genai import Client
import json
import tempfile
import logging
import datetime
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
            'transcript': str - The transcribed text,
            'word_confidences': dict - Map of word to confidence score (0.0-1.0)
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
        enable_word_confidence=True,  # Enable word-level confidence scores
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
                # Extract transcript and word confidence scores
                transcript_parts = []
                word_confidences = {}

                for result in response.results:
                    alternative = result.alternatives[0]
                    transcript_parts.append(alternative.transcript)

                    # Extract word-level confidence scores if available
                    if hasattr(alternative, 'words') and alternative.words:
                        for word_info in alternative.words:
                            word = word_info.word.lower()
                            confidence = word_info.confidence if hasattr(word_info, 'confidence') else 0.5
                            word_confidences[word] = confidence
                            logger.info(f"Word: '{word}' - Confidence: {confidence:.3f}")

                transcript = " ".join(transcript_parts)
                logger.info(f"Inline transcription successful ({len(transcript)} chars): '{transcript[:100]}...'")
                logger.info(f"Extracted {len(word_confidences)} word confidence scores")

                return {
                    'transcript': transcript,
                    'word_confidences': word_confidences
                }
            else:
                logger.warning("No transcription results from inline recognize()")
                return {
                    'transcript': '',
                    'word_confidences': {}
                }

        # For longer audio (>50 seconds at 32kbps), use long_running_recognize() with Cloud Storage
        else:
            logger.info(f"Using long_running_recognize() method (audio size: {audio_size / 1024:.1f} KB > {SIZE_THRESHOLD / 1024:.0f} KB threshold, est. {estimated_duration:.1f}s)")

            if not bucket:
                logger.warning("Bucket not available for long audio transcription, attempting fallback to inline recognize()")
                # Fallback: try inline recognize() even though it might fail for very long audio
                # This is better than returning empty string
                try:
                    audio = speech.RecognitionAudio(content=audio_content)
                    response = client.recognize(config=config, audio=audio)
                    if response.results:
                        # Extract transcript and word confidence scores
                        transcript_parts = []
                        word_confidences = {}

                        for result in response.results:
                            alternative = result.alternatives[0]
                            transcript_parts.append(alternative.transcript)

                            # Extract word-level confidence scores if available
                            if hasattr(alternative, 'words') and alternative.words:
                                for word_info in alternative.words:
                                    word = word_info.word.lower()
                                    confidence = word_info.confidence if hasattr(word_info, 'confidence') else 0.5
                                    word_confidences[word] = confidence

                        transcript = " ".join(transcript_parts)
                        logger.info(f"Fallback inline transcription successful: '{transcript}'")
                        return {
                            'transcript': transcript,
                            'word_confidences': word_confidences
                        }
                    else:
                        logger.error("Fallback inline transcription returned no results")
                        return {
                            'transcript': '',
                            'word_confidences': {}
                        }
                except Exception as fallback_error:
                    logger.error(f"Fallback inline transcription failed: {str(fallback_error)}")
                    return {
                        'transcript': '',
                        'word_confidences': {}
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
                # Extract transcript and word confidence scores
                transcript_parts = []
                word_confidences = {}

                for result in response.results:
                    alternative = result.alternatives[0]
                    transcript_parts.append(alternative.transcript)

                    # Extract word-level confidence scores if available
                    if hasattr(alternative, 'words') and alternative.words:
                        for word_info in alternative.words:
                            word = word_info.word.lower()
                            confidence = word_info.confidence if hasattr(word_info, 'confidence') else 0.5
                            word_confidences[word] = confidence
                            logger.info(f"Word: '{word}' - Confidence: {confidence:.3f}")

                transcript = " ".join(transcript_parts)
                logger.info(f"Long-running transcription successful ({len(transcript)} chars): '{transcript[:100]}...'")
                logger.info(f"Extracted {len(word_confidences)} word confidence scores")

                return {
                    'transcript': transcript,
                    'word_confidences': word_confidences
                }
            else:
                logger.warning("No transcription results from long_running_recognize()")
                return {
                    'transcript': '',
                    'word_confidences': {}
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
                'word_confidences': {}
            }

        logger.error(f"Error during transcription: {str(e)}")

        # For longer audio that failed, try fallback to standard recognize if size permits
        if audio_size > SIZE_THRESHOLD and audio_size <= 10 * 1024 * 1024:
            logger.info("Attempting fallback to standard recognize()")
            try:
                audio_inline = speech.RecognitionAudio(content=audio_content)
                response = client.recognize(config=config, audio=audio_inline)
                if response.results:
                    # Extract transcript and word confidence scores
                    transcript_parts = []
                    word_confidences = {}

                    for result in response.results:
                        alternative = result.alternatives[0]
                        transcript_parts.append(alternative.transcript)

                        # Extract word-level confidence scores if available
                        if hasattr(alternative, 'words') and alternative.words:
                            for word_info in alternative.words:
                                word = word_info.word.lower()
                                confidence = word_info.confidence if hasattr(word_info, 'confidence') else 0.5
                                word_confidences[word] = confidence

                    transcript = " ".join(transcript_parts)
                    logger.info(f"Fallback transcription successful: '{transcript}'")
                    return {
                        'transcript': transcript,
                        'word_confidences': word_confidences
                    }
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")

        return {
            'transcript': '',
            'word_confidences': {}
        }

# Calculate pronunciation score when doing free speech
def assess_free_speech(transcription_data):
    """
    Evaluate pronunciation for free speech mode

    Args:
        transcription_data: Dict with 'transcript' and 'word_confidences'
    """
    transcript = transcription_data.get('transcript', '')
    word_confidences = transcription_data.get('word_confidences', {})
    return pronunciation_assessment(transcript, word_confidences)

# Calculate pronunciation score when practicing with reference phrases
def assess_practice_phrase(transcription_data, reference_level):
    """
    Evaluate pronunciation with reference to a specific practice phrase

    Args:
        transcription_data: Dict with 'transcript' and 'word_confidences'
        reference_level: Level of reference phrase (short, medium, extended)
    """
    transcript = transcription_data.get('transcript', '')
    word_confidences = transcription_data.get('word_confidences', {})

    if reference_level not in REFERENCES:
        return pronunciation_assessment(transcript, word_confidences)

    reference_text = REFERENCES[reference_level]

    # Compare transcribed text with reference text
    similarity_score = fuzz.token_sort_ratio(transcript.lower(), reference_text.lower())

    # Get a base assessment using confidence scores
    base_assessment = pronunciation_assessment(transcript, word_confidences)

    # Adjust score based on similarity to reference
    similarity_bonus = (similarity_score - 60) * 0.2 if similarity_score > 60 else 0
    adjusted_score = min(100, base_assessment["score"] + similarity_bonus)

    # Create a new assessment with adjusted scores
    assessment = {
        "score": round(adjusted_score, 1),
        "level": base_assessment["level"],
        "reference_text": reference_text,
        "similarity": similarity_score,
        "feedback": base_assessment["feedback"],
        "strengths": base_assessment["strengths"],
        "areas_for_improvement": base_assessment["areas_for_improvement"]
    }

    # Add reference-specific feedback
    if similarity_score < 50:
        assessment["areas_for_improvement"].insert(0, "Your response differed significantly from the reference phrase")
    elif similarity_score < 75:
        assessment["areas_for_improvement"].insert(0, "Try to follow the reference phrase more closely")
    else:
        assessment["strengths"].insert(0, "Good reproduction of the reference phrase")

    return assessment

# Calculate pronunciation score based on clarity metrics
def pronunciation_assessment(transcribed_text, word_confidences=None):
    """
    Evaluate pronunciation using clarity metrics:
    - Accuracy: Based on Google Speech-to-Text confidence scores (how clearly words were pronounced)
    - Recognition: How well are words recognized?
    - Text complexity: Can they produce appropriate sentence structures?
    - Vocabulary variety: Range of vocabulary used

    Args:
        transcribed_text: The transcribed text
        word_confidences: Dict mapping words to confidence scores (0.0-1.0)
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

    # Use word confidence scores from Google Speech-to-Text
    # This represents how clearly each word was pronounced (0.0-1.0)
    word_scores = []
    mispronounced_words = []

    if word_confidences is None:
        word_confidences = {}

    for word in words:
        word_clean = word.lower().strip('.,!?;:')

        # Get confidence score from Google Speech-to-Text
        # Convert to 0-100 scale for internal scoring
        if word_clean in word_confidences:
            confidence = word_confidences[word_clean]
            score = confidence * 100
            logger.info(f"Word '{word_clean}' - Confidence: {confidence:.3f} (score: {score:.1f})")
        else:
            # Fallback: if confidence not available, use moderate score
            # This can happen with punctuation or short words
            score = 70
            logger.info(f"Word '{word_clean}' - No confidence available, using default: {score}")

        # Mark words with low confidence as mispronounced
        if score < 80:
            mispronounced_words.append(word_clean)

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

def generate_tts_feedback(text, level):
    """Generate Text-to-Speech audio feedback in Spanish"""
    try:
        # Initialize Text-to-Speech client
        client = texttospeech.TextToSpeechClient()
        
        # Select voice based on clarity profile (slower for initial clarity)
        speaking_rate = 0.8 if "Profile A" in level else 1.0
        
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
        
        # Process in memory
        audio_content = file.read()
        logger.info(f"Received audio file of size: {len(audio_content)} bytes")
        
        # Transcribe audio
        transcription_data = transcribe_audio(audio_content)
        spoken_text = transcription_data.get('transcript', '')
        word_confidences = transcription_data.get('word_confidences', {})

        if not spoken_text:
            logger.warning("No transcription returned")
            return jsonify({
            "score": 70,
            "level": "Profile B - Functional Clarity",
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
            assessment = assess_practice_phrase(transcription_data, practice_level)
            corrected_text = REFERENCES[practice_level]  # Use reference as corrected text
            logger.info(f"Practice mode assessment: level={practice_level}, score={assessment['score']}")
        else:
            # Free speech mode
            assessment = assess_free_speech(transcription_data)
            corrected_text = generate_corrected_text(spoken_text)
            logger.info(f"Free speech assessment: level={assessment['level']}, score={assessment['score']}")

        # Generate TTS feedback
        tts_url = generate_tts_feedback(corrected_text, assessment['level'])

        # Prepare response
        response = {
            "score": assessment['score'],
            "level": assessment['level'],
            "transcribed_text": spoken_text,
            "corrected_text": corrected_text,
            "feedback": assessment['feedback'],
            "strengths": assessment['strengths'],
            "areas_for_improvement": assessment['areas_for_improvement'],
            "tts_audio_url": tts_url
        }
        
        # Add practice-specific fields if applicable
        if practice_level and 'reference_text' in assessment:
            response["reference_text"] = assessment['reference_text']
            response["similarity"] = assessment['similarity']
        
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

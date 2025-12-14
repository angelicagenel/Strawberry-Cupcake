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
                "beginner": "Hola, ¿cómo estás? Espero que estés teniendo un buen día.",
                "intermediate": "Los bomberos llegaron rápidamente al lugar del incendio.",
                "advanced": "En caso de emergencia, mantenga la calma y siga las instrucciones de seguridad."
            }
    except Exception as e:
        logger.error(f"Error loading references: {e}")
        return {
            "beginner": "Hola, ¿cómo estás?",
            "intermediate": "Me gusta viajar y conocer nuevas culturas.",
            "advanced": "La educación es fundamental para el desarrollo de la sociedad."
        }

# Load ACTFL criteria from configuration file
def load_actfl_criteria():
    """Load detailed ACTFL proficiency criteria from configuration file"""
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
                    logger.warning(f"ACTFL criteria file not found in bucket {BUCKET_NAME}")
            # Return None if file not found - will use built-in criteria
            logger.warning("ACTFL criteria file not found. Using built-in criteria.")
            return None
    except Exception as e:
        logger.error(f"Error loading ACTFL criteria: {e}")
        return None

# Initialize Spanish dictionary, references, and ACTFL criteria
SPANISH_DICT = load_dictionary()
REFERENCES = load_references()
ACTFL_CRITERIA = load_actfl_criteria()
logger.info(f"Dictionary loaded with {len(SPANISH_DICT)} words")
logger.info(f"ACTFL criteria loaded: {'Yes' if ACTFL_CRITERIA else 'No (using built-in)'}")

def transcribe_audio(audio_content):
    """Transcribe Spanish audio using Google Cloud Speech-to-Text with support for up to 90 seconds"""
    client = speech.SpeechClient()

    # Upload audio to Cloud Storage for long-running recognition
    if bucket:
        blob_name = f"temp_audio/{uuid.uuid4()}.webm"
        blob = bucket.blob(blob_name)
        blob.upload_from_bytes(audio_content)

        # Create GCS URI
        gcs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
        audio = speech.RecognitionAudio(uri=gcs_uri)
    else:
        # Fallback to inline audio (works up to 60 seconds)
        audio = speech.RecognitionAudio(content=audio_content)

    # Configuration for long audio
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
        sample_rate_hertz=48000,
        language_code="es-ES",
        alternative_language_codes=["es-MX", "es-US"],
        enable_automatic_punctuation=True,
        use_enhanced=True,
        model="default",
        audio_channel_count=1
    )

    try:
        # Use long_running_recognize for audio up to 90 seconds
        operation = client.long_running_recognize(config=config, audio=audio)

        # Wait for operation to complete (timeout 120 seconds to allow processing)
        response = operation.result(timeout=120)

        # Clean up temporary file if uploaded to bucket
        if bucket:
            try:
                blob.delete()
            except:
                pass

        if response.results:
            transcript = " ".join(result.alternatives[0].transcript for result in response.results)
            logger.info(f"Transcription successful: '{transcript}'")
            return transcript
        else:
            logger.warning("No transcription results")
            return ""

    except Exception as e:
        logger.error(f"Error in long_running_recognize: {str(e)}")

        # Fallback to standard recognize for shorter audio
        try:
            audio_inline = speech.RecognitionAudio(content=audio_content)
            response = client.recognize(config=config, audio=audio_inline)
            if response.results:
                transcript = " ".join(result.alternatives[0].transcript for result in response.results)
                return transcript
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {str(fallback_error)}")

        return ""

# Calculate pronunciation score when doing free speech
def assess_free_speech(transcribed_text):
    """
    Evaluate pronunciation using ACTFL FACT criteria for free speech mode
    """
    return actfl_assessment(transcribed_text)

# Calculate pronunciation score when practicing with reference phrases
def assess_practice_phrase(transcribed_text, reference_level):
    """
    Evaluate pronunciation with reference to a specific practice phrase
    """
    if reference_level not in REFERENCES:
        return actfl_assessment(transcribed_text)
    
    reference_text = REFERENCES[reference_level]
    
    # Compare transcribed text with reference text
    similarity_score = fuzz.token_sort_ratio(transcribed_text.lower(), reference_text.lower())
    
    # Get a base assessment
    base_assessment = actfl_assessment(transcribed_text)
    
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

# Calculate pronunciation score based on ACTFL FACT criteria
def actfl_assessment(transcribed_text):
    """
    Evaluate pronunciation using ACTFL FACT criteria:
    - Functions and tasks: Can the speaker communicate their message?
    - Accuracy: How precise is their pronunciation?
    - Context and content: Can they handle the topic appropriately?
    - Text type: Can they produce appropriate sentence structures?
    """
    words = transcribed_text.split()
    if not words:
        logger.warning("No words to score")
        return {
            "score": 70.0,
            "level": "Novice Mid",
            "feedback": "We couldn't detect your speech. Please ensure your microphone is working and try speaking a bit louder. Keep going!",
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
    
    # Calculate composite ACTFL score with different weights
    # Pronunciation accuracy is most important
    composite_score = (
        accuracy_score * 0.6 +
        recognized_word_percentage * 100 * 0.2 +
        text_complexity * 0.1 +
        vocabulary_score * 0.1
    )
    
    # Native speaker adjustment - boost scores for clearly native speakers
    if accuracy_score > 90 and len(words) > 3:
        composite_score = min(100, composite_score + 5)
    
    # Determine ACTFL level
    level = determine_actfl_level(composite_score, len(words), recognized_word_percentage)
    
    # Generate feedback
    strengths = generate_strengths(accuracy_score, recognized_word_percentage, len(words))
    areas_for_improvement = generate_improvements(mispronounced_words, accuracy_score)
    
    logger.info(f"ACTFL Scoring - Accuracy: {accuracy_score}, Recognition: {recognized_word_percentage*100}, " +
              f"Text: {text_complexity}, Vocab: {vocabulary_score}, Final: {composite_score}, Level: {level}")

    # Build the assessment result
    assessment_result = {
        "score": round(composite_score, 1),
        "level": level,
        "feedback": generate_feedback(level),
        "strengths": strengths,
        "areas_for_improvement": areas_for_improvement,
        "word_scores": dict(zip(words, word_scores))
    }

    # Add detailed criteria descriptors if available
    if ACTFL_CRITERIA:
        for level_key, criteria in ACTFL_CRITERIA.items():
            if criteria["name"] == level:
                assessment_result["criteria_details"] = {
                    "oral_production": criteria["oral_production"],
                    "functions": criteria["functions"],
                    "discourse": criteria["discourse"],
                    "grammatical_control": criteria["grammatical_control"],
                    "vocabulary": criteria["vocabulary"],
                    "pronunciation": criteria["pronunciation"],
                    "communication_strategies": criteria["communication_strategies"],
                    "sociocultural_use": criteria["sociocultural_use"]
                }
                break

    return assessment_result

def determine_actfl_level(score, word_count, recognized_ratio):
    """Determine ACTFL proficiency level based on score and other factors"""

    # If we have loaded ACTFL criteria, use the score ranges from there
    if ACTFL_CRITERIA:
        # Adjust score based on performance factors
        adjusted_score = score

        # Boost score for longer, more complex speech (discourse production)
        if word_count >= 15 and recognized_ratio >= 0.9:
            adjusted_score = min(100, score + 3)
        elif word_count >= 10 and recognized_ratio >= 0.85:
            adjusted_score = min(100, score + 2)
        elif word_count >= 5 and recognized_ratio >= 0.7:
            adjusted_score = min(100, score + 1)

        # Penalize for very short utterances or low recognition
        if word_count < 3 or recognized_ratio < 0.5:
            adjusted_score = max(0, score - 5)

        # Find the matching level based on score range
        for level_key, criteria in ACTFL_CRITERIA.items():
            score_min, score_max = criteria["score_range"]
            if score_min <= adjusted_score <= score_max:
                return criteria["name"]

        # Fallback to Novice Low if no match found
        return "Novice Low"

    # Fallback to original logic if criteria file not loaded
    # Advanced Levels (Score 80-100)
    if score >= 80:
        if score >= 90:
            # Advanced High: High score + strong performance on length and clarity
            if recognized_ratio >= 0.9 and word_count >= 15:
                return "Advanced High"
            # Advanced Low/Mid combined: Excellent score, but maybe less extended discourse
            else:
                return "Advanced Mid"

        elif score >= 85:
             # Advanced Low: Good score, but needs longer discourse/fewer errors for High
            if recognized_ratio >= 0.85 and word_count >= 10:
                return "Advanced Low"
            else:
                return "Intermediate High"
        else:
            return "Intermediate High"

    # Intermediate Levels (Score 65-79)
    elif score >= 65:
        if score >= 75:
            # Intermediate High: Can produce connected discourse (paragraphs)
            if word_count >= 8 and recognized_ratio >= 0.8:
                return "Intermediate High"
            else:
                return "Intermediate Mid"

        elif score >= 70:
            # Intermediate Mid: Handles straightforward situations (survival level)
            if recognized_ratio >= 0.7 and word_count >= 5:
                return "Intermediate Mid"
            else:
                return "Intermediate Low"
        else:
            return "Intermediate Low"

    # Novice Levels (Score 50-64)
    elif score >= 50:
        if score >= 60:
            # Novice High: Short, predictable messages/simple sentences
            if word_count >= 3 and recognized_ratio >= 0.6:
                return "Novice High"
            else:
                return "Novice Mid"
        elif score >= 55:
            # Novice Mid: Basic needs/personal info with memorized phrases
            return "Novice Mid"
        else:
            return "Novice Low"

    else:
        return "Novice Low"

def generate_feedback(level):
    """Generate feedback text based on ACTFL level"""

    # If we have loaded ACTFL criteria, use feedback templates from there
    if ACTFL_CRITERIA:
        for level_key, criteria in ACTFL_CRITERIA.items():
            if criteria["name"] == level:
                return criteria["feedback_template"]

    # Fallback to built-in feedback templates
    feedback_templates = {
        # Profile C — Consistent Clarity (80-100)
        "Distinguished": "Your pronunciation is clear and generally consistent. Small refinements will help improve overall naturalness and ease of understanding.",
        "Superior": "Your pronunciation is clear and generally consistent. Small refinements will help improve overall naturalness and ease of understanding.",
        "Advanced High": "Your pronunciation is clear and generally consistent. Small refinements will help improve overall naturalness and ease of understanding.",
        "Advanced Mid": "Your pronunciation is clear and generally consistent. Small refinements will help improve overall naturalness and ease of understanding.",
        "Advanced Low": "Your pronunciation is clear and generally consistent. Small refinements will help improve overall naturalness and ease of understanding.",

        # Profile B — Functional Clarity (65-79)
        "Intermediate High": "Your pronunciation is developing steadily, and many sounds are coming through clearly. Continued practice will help you gain more stability and confidence.",
        "Intermediate Mid": "Your pronunciation is developing steadily, and many sounds are coming through clearly. Continued practice will help you gain more stability and confidence.",
        "Intermediate Low": "Your pronunciation is developing steadily, and many sounds are coming through clearly. Continued practice will help you gain more stability and confidence.",

        # Profile A — Initial Clarity (0-64)
        "Novice High": "Your pronunciation is still developing, and this attempt reflects early-stage language use. Keep going — consistency builds clarity.",
        "Novice Mid": "Your pronunciation is still developing, and this attempt reflects early-stage language use. Keep going — consistency builds clarity.",
        "Novice Low": "Your pronunciation is still developing, and this attempt reflects early-stage language use. Keep going — consistency builds clarity."
    }
    return feedback_templates.get(level, "Your pronunciation shows varying levels of accuracy.")

def generate_strengths(accuracy, recognition_ratio, word_count):
    """Generate list of strengths based on performance"""
    strengths = []

    # Strengths based on Accuracy
    if accuracy >= 90:
        strengths.append("Most words were pronounced clearly.")
    elif accuracy >= 85:
        strengths.append("Several sounds were produced clearly and consistently.")
    elif accuracy >= 75:
        strengths.append("You maintained clear pronunciation across common sounds.")
    elif accuracy >= 65:
        strengths.append("Your pronunciation supported basic understanding.")

    # Strengths based on Recognition/Clarity
    if recognition_ratio >= 0.9:
        strengths.append("Familiar words were articulated with good clarity.")
    elif recognition_ratio >= 0.8:
        strengths.append("Most words were pronounced clearly.")
    elif recognition_ratio >= 0.6:
        strengths.append("Several sounds were produced clearly and consistently.")

    # Strengths based on Text Type/Length (Extended Speech)
    if word_count >= 15:
        strengths.append("Your speech showed moments of clear, stable pronunciation.")
    elif word_count >= 8:
        strengths.append("You used simple sentences and familiar phrases effectively.")
    elif word_count >= 5:
        strengths.append("You used simple sentences and familiar phrases effectively.")

    if not strengths:
        strengths.append("Your speech showed moments of clear, stable pronunciation.")

    return strengths

def generate_improvements(mispronounced, accuracy):
    """Generate suggested areas for improvement"""
    improvements = []

    # Generic suggestions based on accuracy score
    if accuracy < 60:
        improvements.append("Continue building basic vocabulary and common greetings.")
        improvements.append("Practice maintaining steady pronunciation across short phrases.")
    elif accuracy < 75:
        improvements.append("Focus on keeping each sound clear from the beginning to the end of the word.")
        improvements.append("Practice keeping your intonation steady across short phrases.")
    elif accuracy < 85:
        improvements.append("Work on maintaining consistent clarity as phrases become longer.")
        improvements.append("Focus on letting your voice settle naturally at the end of the phrase.")
    elif accuracy < 95:
        improvements.append("Pay attention to keeping sounds stable, especially in familiar words.")
        improvements.append("Work on maintaining a smooth, even rhythm while speaking.")

    # Specific Mispronunciation Feedback
    if mispronounced:
        for word in mispronounced[:3]:
            improvements.append(f"Specifically target the pronunciation of {word} to improve clarity and reduce ambiguity.")

    # Final suggestion if no other specific improvements were generated
    if not improvements:
        improvements.append("Pay attention to keeping sounds stable, especially in familiar words.")
        improvements.append("Work on maintaining a smooth, even rhythm while speaking.")

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
        
        # Select voice based on proficiency level (slower for beginners)
        speaking_rate = 0.8 if level.startswith("Novice") else 1.0
        
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
        spoken_text = transcribe_audio(audio_content)
        
        if not spoken_text:
            logger.warning("No transcription returned")
            return jsonify({
            "score": 70,
            "level": "Novice Mid",
            "transcribed_text": "No se pudo transcribir el audio. Por favor, intente de nuevo hablando claramente en español.",
            "corrected_text": "No transcription available. Try speaking more slowly and clearly in Spanish.",
            "error": "Could not transcribe audio. Please try again with clearer pronunciation.",
            "feedback": "Our system had difficulty understanding your speech. This could be due to background noise, speaking too quietly, or using vocabulary that's difficult to recognize.",
            "strengths": ["Attempt to speak in Spanish"],
        "areas_for_improvement": [
            "Speak clearly and at a moderate pace", 
            "Use a good quality microphone",
            "Reduce background noise",
            "Try the Beginner prompt first to test your setup"
            ],
            "tts_audio_url": None
        })
        
        # Calculate assessment based on mode
        if practice_level and practice_level in REFERENCES:
            # Practice mode with reference phrase
            assessment = assess_practice_phrase(spoken_text, practice_level)
            corrected_text = REFERENCES[practice_level]  # Use reference as corrected text
            logger.info(f"Practice mode assessment: level={practice_level}, score={assessment['score']}")
        else:
            # Free speech mode
            assessment = assess_free_speech(spoken_text)
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
            "beginner": "Hola, ¿cómo estás?"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

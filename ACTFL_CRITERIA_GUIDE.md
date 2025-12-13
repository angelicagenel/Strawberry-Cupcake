# ACTFL Proficiency Criteria Guide

## Overview

This application uses the **ACTFL (American Council on the Teaching of Foreign Languages) Proficiency Guidelines** to assess Spanish language pronunciation and overall oral proficiency. The system evaluates learners across 11 proficiency levels, from Novice Low to Distinguished.

## ACTFL Proficiency Levels

The system implements the following proficiency levels:

### Novice Levels (Beginner)

1. **Novice Low** (0-54 points)
   - Limited to isolated words and memorized phrases
   - Functions restricted to naming, identifying, greeting
   - Heavy L1 (native language) interference

2. **Novice Mid** (55-59 points)
   - Short phrases and very short sentences
   - Can respond to direct, predictable questions
   - Strong L1 influence requiring sympathetic listener

3. **Novice High** (60-64 points)
   - Simple sentences on familiar topics
   - Basic survival tasks and limited question formation
   - Generally intelligible though non-native

### Intermediate Levels

4. **Intermediate Low** (65-69 points)
   - Sentence strings for everyday transactions
   - Can initiate interaction in familiar contexts
   - Pronunciation intelligible in familiar situations

5. **Intermediate Mid** (70-74 points)
   - Sustained sentence strings with emerging organization
   - Can maintain conversations and ask varied questions
   - Generally clear without listener strain

6. **Intermediate High** (75-79 points)
   - Approaches paragraph-length discourse
   - Narration and description across time frames
   - Fluent with minimal interference

### Advanced Levels

7. **Advanced Low** (80-84 points)
   - Sustained paragraph-level production
   - Consistent narration in past, present, and future
   - Supports comprehension with minimal listener effort

8. **Advanced Mid** (85-89 points)
   - Consistently sustained discourse
   - Handles complications effectively
   - Supports effortless comprehension

9. **Advanced High** (90-94 points)
   - Extended, detailed paragraphs
   - Handles unexpected complications
   - Near-native pronunciation quality

### Superior & Distinguished Levels

10. **Superior** (95-97 points)
    - Extended and abstract discourse
    - Discusses societal issues and hypotheticals
    - Native-like pronunciation quality

11. **Distinguished** (98-100 points)
    - Discourse comparable to educated native speakers
    - Tailors language to audience
    - Complete cultural and linguistic mastery

## Assessment Dimensions

The system evaluates learners across eight key dimensions defined by ACTFL:

### 1. Oral Production
The length and complexity of speech produced (words, phrases, sentences, paragraphs).

### 2. Functions and Tasks
What the speaker can accomplish with language (greeting, describing, narrating, persuading, etc.).

### 3. Discourse
How speech is organized and connected (fragmented, sequential, cohesive, sophisticated).

### 4. Grammatical Control
Accuracy and consistency in using grammar structures.

### 5. Vocabulary
Range, precision, and appropriateness of word choice.

### 6. Pronunciation
Clarity, intelligibility, and native-like quality of speech sounds.

### 7. Communication Strategies
Methods used to maintain communication (repetition, circumlocution, clarification).

### 8. Sociocultural Use
Appropriateness of language use in different social and cultural contexts.

## How Scoring Works

The system calculates a composite score (0-100) based on:

- **Pronunciation Accuracy (60%)**: How well words match Spanish dictionary
- **Recognition Rate (20%)**: Percentage of clearly recognized words
- **Text Complexity (10%)**: Length and sophistication of response
- **Vocabulary Score (10%)**: Variety and uniqueness of words used

### Score Adjustments

The score is adjusted based on:
- **Discourse length**: Longer, coherent responses receive bonuses
- **Recognition clarity**: High word recognition boosts the score
- **Utterance complexity**: Very short or unclear speech is penalized

## Using the Criteria File

The `actfl_criteria.json` file contains detailed descriptors for each proficiency level. The application:

1. Loads criteria at startup from local file or cloud storage
2. Uses score ranges to determine proficiency level
3. Generates feedback using level-specific templates
4. Provides detailed criteria descriptors in assessment results

### Criteria File Structure

```json
{
  "LEVEL_NAME": {
    "name": "Display name",
    "score_range": [min, max],
    "oral_production": "Description...",
    "functions": "Description...",
    "discourse": "Description...",
    "grammatical_control": "Description...",
    "vocabulary": "Description...",
    "pronunciation": "Description...",
    "communication_strategies": "Description...",
    "sociocultural_use": "Description...",
    "feedback_template": "User-facing feedback message..."
  }
}
```

## Practice vs. Free Speech Modes

### Practice Mode
- Compares learner speech to reference phrases
- Adjusts score based on similarity to reference
- Uses reference text as the "corrected" version
- Ideal for guided practice and improvement

### Free Speech Mode
- Evaluates spontaneous speech production
- Uses AI-powered grammar correction
- Focuses on overall proficiency assessment
- Ideal for authentic communication practice

## API Response Format

When assessment includes ACTFL criteria, the response contains:

```json
{
  "score": 85.5,
  "level": "Advanced Mid",
  "feedback": "Your pronunciation is very strong...",
  "strengths": [...],
  "areas_for_improvement": [...],
  "criteria_details": {
    "oral_production": "...",
    "functions": "...",
    "discourse": "...",
    "grammatical_control": "...",
    "vocabulary": "...",
    "pronunciation": "...",
    "communication_strategies": "...",
    "sociocultural_use": "..."
  }
}
```

## Updating Criteria

To modify or update the ACTFL criteria:

1. Edit `actfl_criteria.json` with new descriptors
2. Ensure score ranges don't overlap
3. Update feedback templates as needed
4. Upload to cloud storage if using bucket deployment
5. Restart the application to load new criteria

## References

- [ACTFL Proficiency Guidelines 2012](https://www.actfl.org/resources/actfl-proficiency-guidelines-2012)
- ACTFL Proficiency Guidelines 2024 (PDF included in repository)

## Notes

- The system provides constructive, encouraging feedback at all levels
- Lower proficiency levels receive slower Text-to-Speech playback
- Word-level confidence scores from Google Speech-to-Text inform the assessment
- Speech Adaptation technology improves recognition for non-native speakers

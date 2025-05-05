# Spanish Pronunciation Feedback Tool 🍓⋆.°

An AI-powered web application that provides feedback on Spanish pronunciation using Google Cloud services.

## Features

- Record audio directly in the browser or upload audio files
- Get an ACTFL-aligned pronunciation score
- Receive detailed feedback on your pronunciation
- Listen to correct pronunciation through text-to-speech
- Track your progress over time

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python with Flask
- **AI Services**: 
  - Google Cloud Speech-to-Text for transcription
  - Google Cloud Text-to-Speech for feedback
  - **License**: This project is licensed under the MIT License.
- **Deployment**: Google Cloud Run

## Deployment to Google Cloud Run

### Prerequisites

1. Google Cloud account with billing enabled
2. Google Cloud CLI installed locally (if deploying from local machine)

### Setup for Google Cloud

1. Create a new Google Cloud project or select an existing one
2. Enable the following APIs:
   - Cloud Run API
   - Cloud Build API
   - Speech-to-Text API
   - Text-to-Speech API
   - Cloud Storage API

### Deployment Steps

#### Option 1: Deploy directly from GitHub

1. From the Google Cloud Console, go to Cloud Run
2. Click "Create Service"
3. Select "Continuously deploy from a repository"
4. Connect to this GitHub repository
5. Configure the build:
   - Set the service name (e.g., "strawberry-cupcake")
   - Choose a region
   - Set authentication to "Allow unauthenticated invocations"
   - Set the environment variable `BUCKET_NAME` to a unique name for your storage bucket
6. Click "Create"

#### Option 2: Deploy using Google Cloud CLI

1. Clone this repository
2. Navigate to the repository directory
3. Run the following command:

```bash
gcloud run deploy angelicagenel-strawberry-cupcake \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="BUCKET_NAME=strawberry-cupcake-files"

# Spanish Vowels Module 1 - Setup Guide

This is your customized speech practice app for the "Power of the 5 Mexican Spanish Pure Vowels" course.

## Overview

This app provides:
- Access code authentication (one code per customer)
- 4 exercises with 2 attempts each
- Progress tracking (students see their own scores)
- Vowel-focused feedback
- Google Sheets tracking (you see all data)

---

## Step 1: Set Up Google Sheets

1. **Create a new Google Sheet**

2. **Create two sheets (tabs):**

   **Tab 1: `access_codes`**
   | code | created_date | notes |
   |------|--------------|-------|
   | VOWEL-TEST1 | 2024-01-15 | Beta tester 1 |
   | VOWEL-ABC2 | 2024-01-15 | Beta tester 2 |

   **Tab 2: `submissions`**
   | timestamp | access_code | exercise | attempt | score | duration_seconds | transcript | source | cohort |
   |-----------|-------------|----------|---------|-------|------------------|------------|--------|--------|

3. **Add your beta tester codes** to the `access_codes` sheet

---

## Step 2: Deploy Google Apps Script

1. In your Google Sheet, go to **Extensions > Apps Script**

2. Delete any existing code and paste the contents of `google_apps_script.js`

3. Save the project (give it a name like "Vowels Module Tracking")

4. Click **Deploy > New deployment**

5. Settings:
   - Type: **Web app**
   - Execute as: **Me**
   - Who has access: **Anyone**

6. Click **Deploy** and authorize when prompted

7. **Copy the Web App URL** - you'll need this for the environment variable

---

## Step 3: Environment Variables

Set these environment variables for deployment:

```bash
# Required - Your existing variables
BUCKET_NAME=your-gcs-bucket
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
GEMINI_API_KEY=your-gemini-api-key

# New - Google Sheets API
SHEETS_API_URL=https://script.google.com/macros/s/YOUR_DEPLOYMENT_ID/exec

# Optional - Legacy webhook (can be same as SHEETS_API_URL or different)
TRACKING_WEBHOOK_URL=https://script.google.com/macros/s/YOUR_DEPLOYMENT_ID/exec
```

---

## Step 4: Test Locally

```bash
cd /home/user/Spanish-Vowels-Module1
export SHEETS_API_URL="your-apps-script-url"
python app.py
```

Open http://localhost:8080 and test with a code from your sheet.

---

## Step 5: Deploy to Google Cloud Run

1. **Build and push the container:**
   ```bash
   gcloud builds submit --config cloudbuild.yaml
   ```

2. **Deploy with environment variables:**
   ```bash
   gcloud run deploy spanish-vowels-m1 \
     --image gcr.io/YOUR_PROJECT/spanish-vowels-m1 \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars="SHEETS_API_URL=your-apps-script-url,BUCKET_NAME=your-bucket"
   ```

---

## How to Use in Rise 360

### Embedding the App

In Rise 360, add a "Multimedia" block and embed using iframe:

```html
<iframe src="https://your-cloud-run-url.run.app/?exercise=L1"
        width="100%"
        height="800"
        frameborder="0">
</iframe>
```

### Different Exercises

Each "RECORD HERE" button should link to the appropriate exercise:

| Lesson | Exercise | URL Parameter |
|--------|----------|---------------|
| Focused Listening | Vowel Intro | `?exercise=L1` |
| Say It Out Loud | Shadowing | `?exercise=L2` |
| Say It Out Loud | Mini-Script | `?exercise=L3` |
| Final Challenge | Final | `?exercise=L4` |

---

## Generating Access Codes

### Option A: Manual
Add codes directly to the Google Sheet `access_codes` tab.

### Option B: Script
In Apps Script, run the `generateAccessCodes` function:
1. Open Apps Script editor
2. In the code, find `generateAccessCodes(count, prefix)`
3. Run: `generateAccessCodes(10, 'VOWEL')` to create 10 codes
4. Check the Logs (View > Logs) to see generated codes

---

## Viewing Student Progress

1. Open your Google Sheet
2. Go to the `submissions` tab
3. Filter by `access_code` to see individual student progress
4. Sort by `timestamp` to see recent activity

### Useful Filters:
- **All submissions for one student:** Filter `access_code` = their code
- **All attempts for one exercise:** Filter `exercise` = L1, L2, etc.
- **Improvement tracking:** Compare Attempt 1 vs Attempt 2 scores

---

## Exercises Configuration

Edit `modules.json` to customize:

```json
{
  "exercises": {
    "L1": {
      "name": "Vowel Intro",
      "prompt": "Your custom instructions...",
      "duration": "15-30 seconds",
      "max_attempts": 2
    }
  }
}
```

---

## Cost Estimate

| Item | Cost per Use |
|------|--------------|
| Speech-to-Text | ~$0.024 per 60-sec recording |
| Text-to-Speech | ~$0.001 per feedback |
| **Total per submission** | **~$0.025** |

**For 10 beta users (8 submissions each):**
- 80 submissions × $0.025 = **~$2.00 total**

---

## Troubleshooting

### "Invalid access code" error
- Check the code exists in `access_codes` sheet
- Verify SHEETS_API_URL is set correctly
- Check Apps Script is deployed and accessible

### "All attempts used" message
- Student has already used 2 attempts for that exercise
- Check `submissions` sheet to verify

### Progress not showing
- Verify SHEETS_API_URL is working
- Check browser console for errors
- Ensure Apps Script has proper permissions

---

## Support

For questions: success@spanish-learning-edge.com

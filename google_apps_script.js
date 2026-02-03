/**
 * Google Apps Script for Spanish Vowels Module 1
 *
 * SETUP INSTRUCTIONS:
 * 1. Create a new Google Sheet
 * 2. Create two sheets (tabs):
 *    - "access_codes" with columns: code, created_date, notes
 *    - "submissions" with columns: timestamp, access_code, exercise, attempt, score, duration_seconds, transcript
 * 3. Go to Extensions > Apps Script
 * 4. Paste this entire code
 * 5. Save and Deploy as Web App:
 *    - Execute as: Me
 *    - Who has access: Anyone
 * 6. Copy the Web App URL and set it as SHEETS_API_URL environment variable
 *
 * SAMPLE access_codes sheet:
 * | code        | created_date | notes       |
 * |-------------|--------------|-------------|
 * | VOWEL-TEST1 | 2024-01-15   | Beta user 1 |
 * | VOWEL-TEST2 | 2024-01-15   | Beta user 2 |
 */

// Configuration - Update with your sheet ID
const SPREADSHEET_ID = SpreadsheetApp.getActiveSpreadsheet().getId();

/**
 * Handle GET requests (validation, progress, attempts)
 */
function doGet(e) {
  const action = e.parameter.action;
  const code = (e.parameter.code || '').toUpperCase().trim();

  let result;

  switch(action) {
    case 'validate':
      result = validateCode(code);
      break;
    case 'progress':
      result = getProgress(code);
      break;
    case 'attempts':
      const exercise = e.parameter.exercise || '';
      result = getAttempts(code, exercise);
      break;
    default:
      result = { error: 'Unknown action' };
  }

  return ContentService
    .createTextOutput(JSON.stringify(result))
    .setMimeType(ContentService.MimeType.JSON);
}

/**
 * Handle POST requests (submit new recording)
 */
function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);

    if (data.action === 'submit') {
      const result = recordSubmission(data);
      return ContentService
        .createTextOutput(JSON.stringify(result))
        .setMimeType(ContentService.MimeType.JSON);
    }

    // Legacy webhook format (for backward compatibility)
    const result = recordSubmission(data);
    return ContentService
      .createTextOutput(JSON.stringify(result))
      .setMimeType(ContentService.MimeType.JSON);

  } catch (error) {
    return ContentService
      .createTextOutput(JSON.stringify({ error: error.message }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

/**
 * Validate an access code
 */
function validateCode(code) {
  if (!code) {
    return { valid: false, message: 'No code provided' };
  }

  const sheet = SpreadsheetApp.openById(SPREADSHEET_ID).getSheetByName('access_codes');
  if (!sheet) {
    return { valid: false, message: 'Configuration error: access_codes sheet not found' };
  }

  const data = sheet.getDataRange().getValues();

  // Skip header row, check if code exists
  for (let i = 1; i < data.length; i++) {
    if (data[i][0].toString().toUpperCase().trim() === code) {
      return { valid: true, message: 'Code validated' };
    }
  }

  return { valid: false, message: 'Invalid code' };
}

/**
 * Get all submissions for a code
 */
function getProgress(code) {
  if (!code) {
    return { submissions: [], message: 'No code provided' };
  }

  const sheet = SpreadsheetApp.openById(SPREADSHEET_ID).getSheetByName('submissions');
  if (!sheet) {
    return { submissions: [], message: 'No submissions found' };
  }

  const data = sheet.getDataRange().getValues();
  const headers = data[0];
  const submissions = [];

  // Find column indices
  const codeCol = headers.indexOf('access_code');
  const exerciseCol = headers.indexOf('exercise');
  const attemptCol = headers.indexOf('attempt');
  const scoreCol = headers.indexOf('score');
  const timestampCol = headers.indexOf('timestamp');

  // Collect submissions for this code
  for (let i = 1; i < data.length; i++) {
    if (data[i][codeCol].toString().toUpperCase().trim() === code) {
      submissions.push({
        exercise: data[i][exerciseCol],
        attempt: parseInt(data[i][attemptCol]) || 1,
        score: parseFloat(data[i][scoreCol]) || 0,
        timestamp: data[i][timestampCol]
      });
    }
  }

  return { submissions: submissions };
}

/**
 * Get attempt count for a specific exercise
 */
function getAttempts(code, exercise) {
  if (!code || !exercise) {
    return { count: 0, max: 2 };
  }

  const sheet = SpreadsheetApp.openById(SPREADSHEET_ID).getSheetByName('submissions');
  if (!sheet) {
    return { count: 0, max: 2 };
  }

  const data = sheet.getDataRange().getValues();
  const headers = data[0];

  const codeCol = headers.indexOf('access_code');
  const exerciseCol = headers.indexOf('exercise');

  let count = 0;

  for (let i = 1; i < data.length; i++) {
    if (data[i][codeCol].toString().toUpperCase().trim() === code &&
        data[i][exerciseCol].toString() === exercise) {
      count++;
    }
  }

  return { count: count, max: 2 };
}

/**
 * Record a new submission
 */
function recordSubmission(data) {
  const sheet = SpreadsheetApp.openById(SPREADSHEET_ID).getSheetByName('submissions');

  if (!sheet) {
    // Create submissions sheet if it doesn't exist
    const ss = SpreadsheetApp.openById(SPREADSHEET_ID);
    const newSheet = ss.insertSheet('submissions');
    newSheet.appendRow(['timestamp', 'access_code', 'exercise', 'attempt', 'score', 'duration_seconds', 'transcript', 'source', 'cohort']);
  }

  const submissionsSheet = SpreadsheetApp.openById(SPREADSHEET_ID).getSheetByName('submissions');

  // Calculate attempt number
  const code = (data.access_code || '').toUpperCase().trim();
  const exercise = data.exercise || '';
  const attemptData = getAttempts(code, exercise);
  const attemptNum = attemptData.count + 1;

  // Append the submission
  submissionsSheet.appendRow([
    data.timestamp || new Date().toISOString(),
    code,
    exercise,
    attemptNum,
    data.score || 0,
    data.duration_seconds || 0,
    data.transcript || '',
    data.source || '',
    data.cohort || ''
  ]);

  return { success: true, attempt: attemptNum };
}

/**
 * Utility: Generate access codes (run manually from script editor)
 */
function generateAccessCodes(count, prefix) {
  const sheet = SpreadsheetApp.openById(SPREADSHEET_ID).getSheetByName('access_codes');
  if (!sheet) {
    Logger.log('Error: access_codes sheet not found');
    return;
  }

  prefix = prefix || 'VOWEL';
  count = count || 10;

  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // Removed confusing chars (0, O, 1, I)

  for (let i = 0; i < count; i++) {
    let code = prefix + '-';
    for (let j = 0; j < 4; j++) {
      code += chars.charAt(Math.floor(Math.random() * chars.length));
    }

    sheet.appendRow([code, new Date().toISOString().split('T')[0], 'Auto-generated']);
    Logger.log('Generated: ' + code);
  }
}

/**
 * Test function - run from script editor to verify setup
 */
function testSetup() {
  // Test code validation
  Logger.log('Testing code validation...');
  const testCode = 'VOWEL-TEST1';
  const validationResult = validateCode(testCode);
  Logger.log('Validation result: ' + JSON.stringify(validationResult));

  // Test progress retrieval
  Logger.log('Testing progress retrieval...');
  const progressResult = getProgress(testCode);
  Logger.log('Progress result: ' + JSON.stringify(progressResult));

  // Test attempts
  Logger.log('Testing attempts...');
  const attemptsResult = getAttempts(testCode, 'L1');
  Logger.log('Attempts result: ' + JSON.stringify(attemptsResult));

  Logger.log('Setup test complete!');
}

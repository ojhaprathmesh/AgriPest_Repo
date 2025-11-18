const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-1.5-flash';
const API_BASE = process.env.API_BASE || 'http://127.0.0.1:5000';
const IDENTIFY_URL = `${API_BASE}/identify`;
const USE_CUSTOM_MODEL_FIRST = true;

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { GEMINI_MODEL, API_BASE, IDENTIFY_URL, USE_CUSTOM_MODEL_FIRST };
}


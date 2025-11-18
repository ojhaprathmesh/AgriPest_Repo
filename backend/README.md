# Custom Model Backend API

This backend serves your trained EfficientNetB0 model and integrates with the Gemini API.

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Update Class Names:**
   - Open `app.py`
   - Update the `CLASS_NAMES` list with your actual model's class names
   - Update the `insect_info` dictionary with information for each class

3. **Update Model Path:**
   - Make sure the `MODEL_PATH` in `app.py` points to your `best_efficientnetb0.h5` file
   - Default path is `../best_efficientnetb0.h5` (one level up from backend folder)

4. **Run the Server:**
   ```bash
   python app.py
   ```
   The server will start on `http://localhost:5000`

## API Endpoints

- `GET /health` - Check if server and model are loaded
- `POST /predict` - Get basic prediction from your model
- `POST /predict-detailed` - Get prediction with detailed information (used by frontend)

## Request Format

```json
{
  "image": "base64_encoded_image_string"
}
```

## Response Format

```json
{
  "name": "Aphid",
  "scientific": "Aphidoidea",
  "family": "Aphididae",
  "habitat": "Crops, gardens, and plants",
  "harmful": "Yes - Harmful",
  "recommendation": "Use insecticidal soap or neem oil",
  "description": "Aphids are small sap-sucking insects...",
  "confidence": 0.95,
  "model_type": "custom_efficientnetb0"
}
```

## Integration

The frontend will:
1. First try your custom model for classification
2. Then use Gemini API for detailed information
3. Combine both results for the best accuracy

If your backend is not running, the frontend will automatically fall back to using Gemini only.


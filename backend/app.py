from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import json
from ai_logic import FeatureExtractor, FuzzyInferer, SpeciesSearcher
from dotenv import load_dotenv
load_dotenv()
from gemini_client import fetch_insect_info

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load your trained model
TORCH_MODEL_PATH = '../models/best_model_torch.pth'
TORCH_MODEL_FALLBACK = '../models/final_model_torch.pth'
CLASS_NAMES_PATH = '../reports/class_names.json'
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INSECT_DB_PATH = 'insect_db.json'
INSECT_DB = {}
FE = FeatureExtractor()
FZ = FuzzyInferer()
SR = SpeciesSearcher()
CLASS_NAMES = []

def load_class_names():
    global CLASS_NAMES
    try:
        p = os.path.join(os.path.dirname(__file__), CLASS_NAMES_PATH)
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                CLASS_NAMES = json.load(f)
            return True
        print(f"Class names file not found at {CLASS_NAMES_PATH}")
        return False
    except Exception as e:
        print(f"Error loading class names: {str(e)}")
        return False

def build_torch_model(num_classes: int):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_feat = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return m

def load_model():
    global model
    if not load_class_names():
        return False
    try:
        path = TORCH_MODEL_PATH if os.path.exists(TORCH_MODEL_PATH) else TORCH_MODEL_FALLBACK
        if not os.path.exists(path):
            print(f"Model file not found at {path}")
            return False
        m = build_torch_model(num_classes=len(CLASS_NAMES))
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        model = m.to(device)
        print(f"Torch model loaded successfully from {path} on {device}")
        return True
    except Exception as e:
        print(f"Error loading torch model: {str(e)}")
        return False

def load_db():
    global INSECT_DB
    try:
        p = os.path.join(os.path.dirname(__file__), INSECT_DB_PATH)
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                INSECT_DB = json.load(f)
            return True
        return False
    except Exception:
        return False

def preprocess_image_torch(image, target_size=(256, 256)):
    try:
        tfm = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = tfm(image).unsqueeze(0).to(device)
        return tensor
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def predict_insect(image):
    try:
        with torch.no_grad():
            inp = preprocess_image_torch(image)
            logits = model(inp)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        predicted_class_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx] if predicted_class_idx < len(CLASS_NAMES) else f"Class_{predicted_class_idx}"
        top_indices = np.argsort(probs)[-3:][::-1]
        top_predictions = [
            {'class': CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}", 'confidence': float(probs[idx])}
            for idx in top_indices
        ]
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'probs': probs.tolist()
        }
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")

def fetch_info(name):
    key = name
    if key in INSECT_DB:
        return INSECT_DB[key]
    return {
        'taxonomy': {'order': 'Unknown', 'family': 'Unknown'},
        'habitat': [],
        'harmfulness': {'label': 'unknown', 'evidence': []},
        'handling': [],
        'description': f'Information not found for {name}',
        'sources': []
    }



@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/report', methods=['GET'])
def report():
    ai_components = {
        'problem_statement': 'Classify insects from images and provide taxonomy, habitat, harmfulness, and handling guidance for agricultural decision support.',
        'state_space_search': {
            'states': 'Candidate species from top-k predictions',
            'initial_state': 'Top-1 predicted species',
            'goal_state': 'Species maximizing combined probability and feature consistency',
            'actions': 'Evaluate next candidate species using heuristic scoring',
            'strategy': 'Best-first search over top-k candidates'
        },
        'knowledge_representation': {
            'method': 'Ontology graph derived from insect_db.json',
            'structure': 'Species -> Family -> Order relationships',
            'implementation': 'OntologyGraph(path) with path() traversal'
        },
        'intelligent_system_design': {
            'components': ['CNN classifier', 'FeatureExtractor', 'FuzzyInferer', 'SpeciesSearcher', 'CSPValidator', 'OntologyGraph'],
            'integration': 'Classifier outputs refined via search, constraints, fuzzy metrics, and ontology retrieval'
        },
        'csp_or_fuzzy_logic': {
            'csp': 'CSPValidator checks feature constraints per species',
            'fuzzy_logic': 'Visibility and harmfulness confidence via FuzzyInferer'
        },
        'other_ai_techniques': ['Transfer learning (EfficientNet/ResNet)', 'Data augmentation'],
        'ethics': {
            'bias': 'Use verified sources, avoid misclassifying beneficial species as harmful',
            'mitigation': 'Low-confidence rejection and source transparency'
        },
        'ai_vs_non_ai': {
            'ai': ['CNN classification', 'best-first search', 'fuzzy inference', 'CSP validation'],
            'non_ai': ['HTTP API', 'JSON formatting', 'file I/O']
        }
    }
    return jsonify(ai_components)



@app.route('/predict-basic', methods=['POST'])
def predict_basic():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        if 'image' in request.json:
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            file = request.files['image']
            image = Image.open(file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        result = predict_insect(image)
        feats = FE.extract(image)
        vis = FZ.visibility(feats)
        refined_name, _, _ = SR.best_first(result['top_predictions'], feats)
        name = refined_name
        info = fetch_info(name)
        harm_label = info.get('harmfulness', {}).get('label', 'unknown')
        harm_conf = FZ.harmfulness_conf(float(result['confidence']), harm_label)
        final_label = 'harmful' if (harm_label == 'harmful' or (harm_label == 'contextual' and harm_conf >= 0.5)) else 'harmless'
        if result['confidence'] < 0.5 or vis < 0.35:
            return jsonify({'error': 'Unidentifiable input'}), 422
        return jsonify({'name': name, 'harmfulness': final_label}), 200
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500

@app.route('/identify', methods=['POST'])
def identify():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        if 'image' in request.json:
            image_data = request.json['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            file = request.files['image']
            image = Image.open(file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        result = predict_insect(image)
        feats = FE.extract(image)
        vis = FZ.visibility(feats)
        refined_name, _, _ = SR.best_first(result['top_predictions'], feats)
        name = refined_name
        info_local = fetch_info(name)
        harm_label = info_local.get('harmfulness', {}).get('label', 'unknown')
        harm_conf = FZ.harmfulness_conf(float(result['confidence']), harm_label)
        final_label = 'harmful' if (harm_label == 'harmful' or (harm_label == 'contextual' and harm_conf >= 0.5)) else 'harmless'
        if result['confidence'] < 0.5 or vis < 0.35:
            return jsonify({'error': 'Unidentifiable input'}), 422
        gem = fetch_insect_info(name, final_label)
        if gem is None:
            resp = {
                'name': name,
                'harmfulness': final_label,
                'family': info_local.get('taxonomy', {}).get('family', 'Unknown'),
                'habitat': info_local.get('habitat', []),
                'recommendation': info_local.get('handling', []),
                'description': info_local.get('description', ''),
                'sources': info_local.get('sources', [])
            }
            return jsonify(resp), 200
        resp = {
            'name': name,
            'harmfulness': final_label,
            'family': gem.get('family'),
            'habitat': gem.get('habitat'),
            'recommendation': gem.get('recommendation'),
            'description': gem.get('description')
        }
        return jsonify(resp), 200
    except Exception as e:
        return jsonify({'error': 'Identification failed', 'message': str(e)}), 500

if __name__ == '__main__':
    print("Loading model...")
    ok_model = load_model()
    print("Loading database...")
    ok_db = load_db()
    if ok_model:
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check the model path.")


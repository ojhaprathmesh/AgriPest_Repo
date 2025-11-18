from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageOps


class FeatureExtractor:
    def extract(self, image: Image.Image) -> Dict[str, float]:
        im = ImageOps.exif_transpose(image).convert('RGB')
        arr = np.asarray(im, dtype=np.float32)
        h, w, _ = arr.shape
        gray = np.mean(arr, axis=2)
        mean_brightness = float(gray.mean())
        contrast = float(gray.std())
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        edges = gx.mean() + gy.mean()
        edge_density = float(edges) / 255.0
        r_mean = float(arr[:, :, 0].mean())
        g_mean = float(arr[:, :, 1].mean())
        b_mean = float(arr[:, :, 2].mean())
        dominant = 'red'
        if g_mean >= r_mean and g_mean >= b_mean:
            dominant = 'green'
        elif b_mean >= r_mean and b_mean >= g_mean:
            dominant = 'blue'
        aspect_ratio = float(max(w, h)) / float(min(w, h))
        return {
            'brightness': mean_brightness,
            'contrast': contrast,
            'edge_density': edge_density,
            'r_mean': r_mean,
            'g_mean': g_mean,
            'b_mean': b_mean,
            'dominant_color': dominant,
            'aspect_ratio': aspect_ratio,
        }


class FuzzyInferer:
    def visibility(self, features: Dict[str, float]) -> float:
        b = features.get('brightness', 127.0) / 255.0
        c = features.get('contrast', 64.0) / 128.0
        b_term = 1.0 - min(1.0, abs(b - 0.5) * 2.0)
        vis = max(0.0, min(1.0, 0.5 * b_term + 0.5 * max(0.0, min(1.0, c))))
        return float(vis)

    def harmfulness_conf(self, base_conf: float, label: str) -> float:
        if label == 'harmful':
            return float(min(1.0, 0.6 * base_conf + 0.4))
        if label == 'beneficial':
            return float(max(0.0, 0.6 * base_conf))
        if label == 'contextual':
            return float(0.5 * base_conf + 0.25)
        return float(0.5 * base_conf)



class CSPValidator:
    def __init__(self):
        self.rules = {
            'Ladybug': lambda f: f.get('dominant_color') == 'red' and 0.8 <= f.get('aspect_ratio', 1.0) <= 1.3 and f.get('contrast', 0.0) > 10.0,
            'Aphid': lambda f: f.get('dominant_color') == 'green' and f.get('aspect_ratio', 1.0) <= 1.4,
            'Caterpillar': lambda f: f.get('aspect_ratio', 1.0) > 1.3,
            'Grasshopper': lambda f: f.get('aspect_ratio', 1.0) > 1.2 and f.get('dominant_color') == 'green',
            'Spider': lambda f: f.get('edge_density', 0.0) > 0.12,
        }

    def satisfies(self, species: str, features: Dict[str, float]) -> bool:
        fn = self.rules.get(species)
        if fn is None:
            return True
        return bool(fn(features))


class SpeciesSearcher:
    def __init__(self):
        self.feature_bonus = {
            'Ladybug': lambda f: 0.15 if f.get('dominant_color') == 'red' else 0.0,
            'Aphid': lambda f: 0.15 if f.get('dominant_color') == 'green' else 0.0,
            'Caterpillar': lambda f: 0.15 if f.get('aspect_ratio', 1.0) > 1.3 else 0.0,
            'Grasshopper': lambda f: 0.12 if (f.get('aspect_ratio', 1.0) > 1.2 and f.get('dominant_color') == 'green') else 0.0,
            'Spider': lambda f: 0.1 if f.get('edge_density', 0.0) > 0.12 else 0.0,
        }
        self.csp = CSPValidator()

    def best_first(self, top: List[Dict], features: Dict[str, float]) -> Tuple[str, float, Dict]:
        scored: List[Tuple[str, float]] = []
        for item in top:
            name = item['class']
            conf = float(item['confidence'])
            bonus_fn = self.feature_bonus.get(name)
            bonus = float(bonus_fn(features)) if bonus_fn else 0.0
            total = conf + bonus
            if not self.csp.satisfies(name, features):
                total *= 0.7
            scored.append((name, total))
        scored.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score = scored[0]
        return best_name, best_score, {'scores': scored, 'features': features}
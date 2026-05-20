import numpy as np
import torch
import torch.nn as nn

class LSTMTemporalModel(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, num_layers=2):
        super(LSTMTemporalModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output)


class TemporalAnalyzer:
    def __init__(self, threshold=0.72 , model_path=None):
        self.threshold = threshold
        self.lstm_model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            self.lstm_model = LSTMTemporalModel()
            self.lstm_model.load_state_dict(
                torch.load(model_path, map_location='cpu')
            )
            self.lstm_model.eval()
            print(f"LSTM model loaded from {model_path}")
        except Exception as e:
            print(f"Model load failed: {e} — using drift analysis")
            self.lstm_model = None

    def compute_feature_drift(self, features):
        if len(features) < 2:
            return [0.0]
        
        drift_scores = []
        for i in range(1, len(features)):
            diff = np.linalg.norm(features[i] - features[i-1])
            drift_scores.append(diff)

        mean_drift = np.mean(drift_scores)
        std_drift = np.std(drift_scores)
        
        normalized = []
        for d in drift_scores:
            if std_drift > 0:
                z = (d - mean_drift) / std_drift
                score = 1 / (1 + np.exp(-z))
            else:
                score = 0.3
            normalized.append(round(float(score), 3))

        return [0.3] + normalized

    def compute_fake_probability(self, features, drift_scores):
        
        if self.lstm_model is not None:
            return self._lstm_predict(features)
        
        
        window = 7
        probs = []
        for i in range(len(drift_scores)):
            start = max(0, i - window // 2)
            end = min(len(drift_scores), i + window // 2 + 1)
            avg = np.mean(drift_scores[start:end])
            probs.append(round(float(avg), 3))
        return probs

    def _lstm_predict(self, features):
        probs = []
        window_size = 10
        
        with torch.no_grad():
            for i in range(len(features)):
                start = max(0, i - window_size + 1)
                window = features[start:i+1]
                
                if len(window) < window_size:
                    padding = np.zeros((window_size - len(window), features.shape[1]))
                    window = np.vstack([padding, window])
                
                tensor = torch.FloatTensor(window).unsqueeze(0)
                prob = self.lstm_model(tensor).item()
                probs.append(round(float(prob), 3))
        
        return probs

    def get_fake_segments(self, probabilities, timestamps):
        segments = []
        in_fake = False
        seg_start = None
        seg_start_idx = None

        for i, (prob, ts) in enumerate(zip(probabilities, timestamps)):
            if prob >= self.threshold and not in_fake:
                in_fake = True
                seg_start = ts
                seg_start_idx = i
            elif prob < self.threshold and in_fake:
                in_fake = False
                seg_probs = probabilities[seg_start_idx:i]
                segments.append({
                    "start": seg_start,
                    "end": timestamps[i-1],
                    "avg_confidence": round(float(np.mean(seg_probs)), 3)
                })

        if in_fake and seg_start_idx is not None:
            seg_probs = probabilities[seg_start_idx:]
            segments.append({
                "start": seg_start,
                "end": timestamps[-1],
                "avg_confidence": round(float(np.mean(seg_probs)), 3)
            })

        return segments
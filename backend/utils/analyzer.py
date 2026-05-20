import numpy as np
import cv2

class WhyFakeEngine:
    def analyze_blink(self, face_frames):
        blink_scores = []
        for face in face_frames:
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            eye_region = gray[int(h*0.2):int(h*0.5), :]
            variance = np.var(eye_region) / 1000.0
            score = min(1.0, variance / 50.0)
            blink_scores.append(score)
        if not blink_scores:
            return 0.3
        avg = np.mean(blink_scores)
        anomaly = abs(avg - 0.5) * 2
        return round(float(anomaly), 3)

    def analyze_lip_sync(self, face_frames):
        mouth_variances = []
        for face in face_frames:
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            mouth_region = gray[int(h*0.65):int(h*0.95), int(w*0.25):int(w*0.75)]
            mouth_variances.append(np.var(mouth_region))
        if len(mouth_variances) < 2:
            return 0.3
        diffs = [abs(mouth_variances[i] - mouth_variances[i-1])
                 for i in range(1, len(mouth_variances))]
        max_diff = max(diffs) if diffs else 1
        normalized = [d / max_diff for d in diffs]
        score = np.mean(normalized)
        return round(float(min(score * 1.2, 1.0)), 3)

    def analyze_feature_drift(self, features):
        if len(features) < 2:
            return 0.3
        drifts = []
        for i in range(1, len(features)):
            drift = np.linalg.norm(features[i] - features[i-1])
            drifts.append(drift)
        if not drifts:
            return 0.3
        max_d = max(drifts)
        if max_d > 0:
            normalized = [d / max_d for d in drifts]
        else:
            normalized = drifts
        score = np.mean(normalized)
        return round(float(min(score * 1.0, 1.0)), 3)

    def get_reasons(self, blink_score, lip_score, drift_score):
        reasons = []
        if blink_score > 0.6:
            reasons.append("Abnormal blink pattern detected")
        if lip_score > 0.6:
            reasons.append("Lip-sync inconsistency found")
        if drift_score > 0.6:
            reasons.append("Facial feature drift detected")
        if not reasons:
            reasons.append("Minor temporal inconsistencies found")
        return reasons

    def compute_authenticity_score(self, blink, lip, drift,
                                    fake_segments, total_duration):
        fake_duration = sum(
            s["end"] - s["start"] for s in fake_segments
        )
        time_ratio = fake_duration / total_duration if total_duration > 0 else 0

        if not fake_segments:
            
            anomaly = (blink * 0.15 + lip * 0.15 + drift * 0.15) 
            final_score = max(75.0, 96.0 - anomaly * 20)
        else:
            anomaly= (blink * 0.3 + lip * 0.3 + drift * 0.4)
            final = (anomaly * 0.4 + time_ratio * 0.6)
            final_score = min(35.0, (1 - final) * 100)
            

        authenticity = round(final_score, 1)

        if authenticity < 40:
            risk = "high"
            color = "red"
        elif authenticity < 70:
            risk = "medium"
            color = "yellow"
        else:
            risk = "low"
            color = "green"

        return {
            "score": authenticity,
            "risk": risk,
            "color": color
        }
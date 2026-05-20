from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import shutil
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.frame_extractor import extract_frames
from utils.face_detector import process_frames_for_faces
from utils.feature_extractor import FeatureExtractor
from utils.temporal_model import TemporalAnalyzer, LSTMTemporalModel
from utils.analyzer import WhyFakeEngine
from utils.gradcam import GradCAM
from utils.report_generator import generate_pdf_report

import numpy as np
import cv2
import base64

app = FastAPI(title="DeepTrace API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pth')
feature_extractor = FeatureExtractor(model_path=model_path)
temporal_analyzer = TemporalAnalyzer(threshold=0.65)
why_fake_engine = WhyFakeEngine()
gradcam = GradCAM()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"status": "DeepTrace backend running"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    print("\n========== NEW REQUEST ==========")
    print(f"[INFO] File received: {file.filename}")

    video_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"[INFO] File saved at: {video_path}")

    frames, timestamps = extract_frames(video_path, fps=10)
    print(f"[INFO] Frames extracted: {len(frames)}")

    if len(frames) == 0:
        print("[ERROR] No frames extracted")
        return {
            "model_prediction": {"verdict": "ERROR", "fake_probability": 0},
            "authenticity": {"score": 0, "risk": "error", "color": "gray"},
            "reasons": ["Video could not be processed"],
            "fake_segments": [],
            "confidence_over_time": [],
            "breakdown": {"blink": 0, "lip_sync": 0, "feature_drift": 0}
        }

    face_frames, valid_indices = process_frames_for_faces(frames)
    print(f"[INFO] Faces detected: {len(face_frames)}")

    if len(face_frames) == 0:
        print("[ERROR] No faces detected")
        return {"error": "No faces detected in video"}

    valid_timestamps = [timestamps[i] for i in valid_indices]
    print(f"[INFO] Valid timestamps count: {len(valid_timestamps)}")

    total_duration = valid_timestamps[-1] if valid_timestamps else 1

    print("[INFO] Running model prediction...")
    video_fake_prob = feature_extractor.predict_video(face_frames)
    print(f"[INFO] Fake probability: {video_fake_prob}")

    features = feature_extractor.extract(face_frames)
    print(f"[INFO] Features extracted: {len(features)}")

    drift_scores = temporal_analyzer.compute_feature_drift(features)
    fake_probs = temporal_analyzer.compute_fake_probability(features, drift_scores)
    print(f"[INFO] Timeline computed: {len(fake_probs)} points")

    if video_fake_prob > 0.5:
        fake_segments = temporal_analyzer.get_fake_segments(fake_probs, valid_timestamps)
        if not fake_segments:
            fake_segments = [{
                "start": 0,
                "end": total_duration,
                "avg_confidence": round(video_fake_prob, 3)
            }]
    else:
        fake_probs = [video_fake_prob] * len(valid_timestamps)
        fake_segments = []

    blink_score = why_fake_engine.analyze_blink(face_frames)
    lip_score = why_fake_engine.analyze_lip_sync(face_frames)
    drift_score = why_fake_engine.analyze_feature_drift(features)

    if video_fake_prob > 0.5:
        reasons = why_fake_engine.get_reasons(blink_score, lip_score, drift_score)
        authenticity_score = round((1 - video_fake_prob) * 100, 1)
        authenticity_score = max(10.0, min(45.0, authenticity_score))
        risk = "high"
        color = "red"
    else:
        reasons = ["No manipulation detected", "Video appears authentic"]
        authenticity_score = round((1 - video_fake_prob) * 100, 1)
        authenticity_score = max(75.0, min(99.0, authenticity_score))
        risk = "low"
        color = "green"

    authenticity = {
        "score": authenticity_score,
        "risk": risk,
        "color": color
    }

    heatmap_b64 = None
    if fake_segments and len(face_frames) > 0:
        print("[INFO] Generating heatmap...")
        mid_idx = len(face_frames) // 2
        heatmap_frame = gradcam.generate_heatmap(face_frames[mid_idx])
        _, buffer = cv2.imencode('.jpg',
            cv2.cvtColor(heatmap_frame, cv2.COLOR_RGB2BGR))
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')

    result = {
        "filename": file.filename,
        "total_frames_analyzed": len(face_frames),
        "duration": total_duration,
        "model_prediction": {
            "fake_probability": round(video_fake_prob, 3),
            "verdict": "FAKE" if video_fake_prob > 0.5 else "REAL"
        },
        "authenticity": authenticity,
        "fake_segments": fake_segments,
        "confidence_over_time": [
            {"timestamp": t, "probability": p}
            for t, p in zip(valid_timestamps, fake_probs)
        ],
        "breakdown": {
            "blink": blink_score,
            "lip_sync": lip_score,
            "feature_drift": drift_score
        },
        "reasons": reasons,
        "heatmap": heatmap_b64
    }

    print("[INFO] Analysis complete ✅")
    print("================================\n")

    return result

@app.post("/report")
async def download_report(file: UploadFile = File(...)):
    video_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    frames, timestamps = extract_frames(video_path, fps=10)
    face_frames, valid_indices = process_frames_for_faces(frames)
    valid_timestamps = [timestamps[i] for i in valid_indices]
    total_duration = valid_timestamps[-1] if valid_timestamps else 1

    video_fake_prob = feature_extractor.predict_video(face_frames)
    features = feature_extractor.extract(face_frames)
    drift_scores = temporal_analyzer.compute_feature_drift(features)
    fake_probs = temporal_analyzer.compute_fake_probability(features, drift_scores)

    if video_fake_prob > 0.5:
        fake_segments = temporal_analyzer.get_fake_segments(fake_probs, valid_timestamps)
        if not fake_segments:
            fake_segments = [{
                "start": 0,
                "end": total_duration,
                "avg_confidence": round(video_fake_prob, 3)
            }]
        reasons = why_fake_engine.get_reasons(
            why_fake_engine.analyze_blink(face_frames),
            why_fake_engine.analyze_lip_sync(face_frames),
            why_fake_engine.analyze_feature_drift(features)
        )
        authenticity_score = round((1 - video_fake_prob) * 100, 1)
        authenticity_score = max(10.0, min(45.0, authenticity_score))
        risk = "high"
        color = "red"
    else:
        fake_segments = []
        reasons = ["No manipulation detected", "Video appears authentic"]
        authenticity_score = round((1 - video_fake_prob) * 100, 1)
        authenticity_score = max(75.0, min(99.0, authenticity_score))
        risk = "low"
        color = "green"

    analysis_result = {
        "authenticity": {"score": authenticity_score, "risk": risk, "color": color},
        "fake_segments": fake_segments,
        "breakdown": {
            "blink": why_fake_engine.analyze_blink(face_frames),
            "lip_sync": why_fake_engine.analyze_lip_sync(face_frames),
            "feature_drift": why_fake_engine.analyze_feature_drift(features)
        },
        "reasons": reasons
    }

    pdf_buffer = generate_pdf_report(analysis_result)
    os.remove(video_path)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=DeepTrace_Report.pdf"
        }
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
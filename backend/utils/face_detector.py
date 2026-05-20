import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

def detect_and_crop_face(frame, target_size=(224, 224)):
    results = detector.detect_faces(frame)
    
    if not results:
        return None
    
    best = max(results, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = best['box']
    
    
    x, y = max(0, x), max(0, y)
    
    face = frame[y:y+h, x:x+w]
    
    if face.size == 0:
        return None
    
    face_resized = cv2.resize(face, target_size)
    return face_resized

def process_frames_for_faces(frames):
    face_frames = []
    valid_indices = []
    
    for i, frame in enumerate(frames):
        face = detect_and_crop_face(frame)
        if face is not None:
            face_frames.append(face)
            valid_indices.append(i)
    
    return face_frames, valid_indices
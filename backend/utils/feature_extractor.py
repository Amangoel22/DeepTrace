import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import numpy as np
import os

class ResNextLSTM(nn.Module):
    def __init__(self, hidden_size=512, num_layers=2, dropout=0.4):
        super(ResNextLSTM, self).__init__()
        
        resnext = timm.create_model(
            'resnext50_32x4d',
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )
        self.cnn = resnext
        
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch, frames, C, H, W = x.shape
        x = x.view(batch * frames, C, H, W)
        features = self.cnn(x)
        features = features.view(batch, frames, -1)
        lstm_out, _ = self.lstm(features)
        last = lstm_out[:, -1, :]
        return self.classifier(last)


class FeatureExtractor:
    def __init__(self, model_path=None):
        self.model = ResNextLSTM()
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location='cpu')
            )
            print("Trained model loaded!")
        else:
            print("No trained model found — using base features")
        
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract(self, face_frames):
        features = []
        with torch.no_grad():
            for face in face_frames:
                tensor = self.transform(face).unsqueeze(0)
                feat = self.model.cnn(tensor)
                features.append(feat.squeeze().numpy())
        return np.array(features)
    
    def predict_video(self, face_frames):
        if len(face_frames) == 0:
            return 0.5
        
        tensors = []
        for face in face_frames:
            tensor = self.transform(face)
            tensors.append(tensor)
        
        while len(tensors) < 10:
            tensors.append(tensors[-1])
        tensors = tensors[:10]
        
        batch = torch.stack(tensors).unsqueeze(0)
        
        with torch.no_grad():
            prob = self.model(batch).item()
        
        return prob
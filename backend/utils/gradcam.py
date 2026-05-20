import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2

class GradCAM:
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        
        
        target_layer = self.model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, face_frame):
        tensor = self.transform(face_frame).unsqueeze(0)
        tensor.requires_grad_(True)
        
        output = self.model(tensor)
        
        
        score = output[0].max()
        self.model.zero_grad()
        score.backward()
        
        if self.gradients is None or self.activations is None:
            return face_frame
        
        
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).numpy()
        
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        #
        h, w = face_frame.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), 
            cv2.COLORMAP_JET
        )
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        
        overlay = cv2.addWeighted(face_frame, 0.6, heatmap_rgb, 0.4, 0)
        return overlay
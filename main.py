from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# here we build the model with the same architecture we trained it on
def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

# load out best saved model
ckpt = torch.load("asl_resnet18_best.pth", map_location=DEVICE)
classes = ckpt["classes"]
IMAGE_SIZE = ckpt.get("image_size", 224)

model = build_model(len(classes)).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet
        std=[0.229, 0.224, 0.225]
    )
])


WINDOW = 10
pred_window = deque(maxlen=WINDOW)

def predict(frame_bgr):
    # Convert BGR -> RGB -> PIL
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    x = preprocess(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)
    return pred.item(), conf.item()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try a different camera index (0,1,2).")

print("Press 'q' to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # crop to center square 
    h, w = frame.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    crop = frame[y0:y0+side, x0:x0+side]

    pred_idx, conf = predict(crop)
    pred_window.append(pred_idx)

    # majority vote smoothing
    smoothed_idx = max(set(pred_window), key=pred_window.count)
    label = classes[smoothed_idx]

    # Confidence threshold
    if conf < 0.60:
        label_to_show = "—"
    else:
        label_to_show = f"{label} ({conf:.2f})"

    # Draw UI
    cv2.rectangle(frame, (x0, y0), (x0+side, y0+side), (0, 255, 0), 2)
    cv2.putText(frame, label_to_show, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)

    cv2.imshow("ASL Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


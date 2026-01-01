from collections import deque

import cv2
import mediapipe as mp
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

@torch.no_grad()
def predict_pil(pil_img):
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    conf, idx = torch.max(probs, dim=0)
    return classes[idx.item()], float(conf.item())

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def landmarks_to_bbox(landmarks, w, h, pad=0.25):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # convert to pixel coords
    x1 = int(x1 * w); x2 = int(x2 * w)
    y1 = int(y1 * h); y2 = int(y2 * h)

    # pad
    bw = x2 - x1
    bh = y2 - y1
    x1 = int(x1 - pad * bw)
    x2 = int(x2 + pad * bw)
    y1 = int(y1 - pad * bh)
    y2 = int(y2 + pad * bh)

    x1 = clamp(x1, 0, w-1)
    x2 = clamp(x2, 0, w-1)
    y1 = clamp(y1, 0, h-1)
    y2 = clamp(y2, 0, h-1)

    return x1, y1, x2, y2

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # load media pipe hands - this will help us focus on the hand
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        display_text = "No hand"
        if result.multi_hand_landmarks:
            hand_lms = result.multi_hand_landmarks[0].landmark
            x1, y1, x2, y2 = landmarks_to_bbox(hand_lms, w, h, pad=0.35)

            # crop and predict
            crop_bgr = frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(crop_rgb)

            pred, conf = predict_pil(pil)
            display_text = f"{pred} ({conf:.2f})"

            # draw bbox + landmarks
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)

        cv2.putText(frame, display_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("ASL V2 - MediaPipe Crop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
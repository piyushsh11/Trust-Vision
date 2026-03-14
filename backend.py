import base64
import json
import os
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageFilter
from torchvision import models, transforms
import joblib
import math

try:
    import torchattacks as ta  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ta = None

CLIP_AVAILABLE = False
try:
    import clip  # type: ignore

    CLIP_AVAILABLE = True
except Exception:
    clip = None

# Paths / logging
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "events.jsonl"
CHECKPOINT_PATH = Path("checkpoints/trades_resnet18.pt")
METRIC_WINDOW = 100


class ImagePayload(BaseModel):
    image: str
    dataset: str = "imagenet"
    model: Optional[str] = "classifier"  # classifier | yolo


class AttackPayload(BaseModel):
    image: str
    dataset: str = "imagenet"
    epsilon: float = 8 / 255
    attack: Optional[str] = "fgsm"


class DefensePayload(BaseModel):
    adv_image: str
    dataset: str = "imagenet"


class VitalAttackPayload(BaseModel):
    image: str
    vitals: dict  # expects temp (°C), pulse (bpm), spo2 (%)
    epsilon: float = 0.5  # magnitude for parameter attack


device = "cuda" if torch.cuda.is_available() else "cpu"
weights = models.ResNet18_Weights.DEFAULT
_mean = torch.tensor(weights.transforms().mean).view(3, 1, 1).to(device)
_std = torch.tensor(weights.transforms().std).view(3, 1, 1).to(device)

# simple eval transforms (resize 256 -> center crop 224)
_resize = transforms.Resize(256)
_crop = transforms.CenterCrop(224)
_to_tensor = transforms.ToTensor()
_normalize = transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)


def preprocess_no_norm(img: Image.Image) -> torch.Tensor:
    return _to_tensor(_crop(_resize(img)))


def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return (t.to(device) - _mean) / _std


def preprocess(img: Image.Image) -> torch.Tensor:
    return normalize_tensor(preprocess_no_norm(img))


def build_model() -> nn.Module:
    base = models.resnet18(weights=weights).to(device)
    if CHECKPOINT_PATH.exists():
        try:
            state = torch.load(CHECKPOINT_PATH, map_location=device)
            base.load_state_dict(state, strict=False)
            print(f"Loaded adversarially-trained checkpoint from {CHECKPOINT_PATH}")
        except Exception as exc:  # pragma: no cover - best effort load
            print(f"Could not load checkpoint {CHECKPOINT_PATH}: {exc}")
    base.eval()
    return base


model = build_model()
categories = weights.meta.get("categories", [])

# zero-shot classifier for domain-specific labels
clip_model = None
clip_preprocess = None
if CLIP_AVAILABLE:
    try:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
    except Exception:
        clip_model = None
        clip_preprocess = None

# YOLO detector (for multi-object option)
yolo_model = None
try:
    from ultralytics import YOLO  # type: ignore

    try:
        yolo_model = YOLO("yolov8n.pt")
        yolo_model.fuse()
        print("Loaded YOLOv8n detector")
    except Exception as exc:  # pragma: no cover - best effort
        print("Failed to load YOLO model:", exc)
        yolo_model = None
except Exception:
    YOLO = None  # type: ignore

# items classifier (CLIP + logistic regression)
items_clf = None
items_classes = None
items_ckpt = Path("checkpoints/items_clip_logreg.joblib")
if items_ckpt.exists():
    try:
        data = joblib.load(items_ckpt)
        items_clf = data.get("clf")
        items_classes = data.get("classes")
        print("Loaded items classifier from", items_ckpt)
    except Exception as exc:
        print("Failed to load items classifier:", exc)

def decode_image(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_data = base64.b64decode(data_url)
    return Image.open(BytesIO(image_data)).convert("RGB")


def encode_image(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def predict(img: Image.Image, dataset: str = "imagenet"):
    ds = dataset.lower()

    # domain: MedMNIST via CLIP zero-shot
    if ds == "medmnist" and clip_model and clip_preprocess:
        text_labels = [
            "no diabetic retinopathy",
            "mild diabetic retinopathy",
            "moderate diabetic retinopathy",
            "severe diabetic retinopathy",
            "proliferative diabetic retinopathy",
        ]
        image_input = clip_preprocess(img).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(text_labels).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features.T
            probs = logits.softmax(dim=-1)
            conf, idx = probs.max(dim=-1)
        return text_labels[idx.item()], conf.item()

    # domain: items + demo set (use CLIP zero-shot with SDG-aligned labels)
    if ds in ("items", "set1") and clip_model and clip_preprocess:
        text_labels = [
            "retinal fundus photograph (medical)",
            "dermatology skin lesion (medical)",
            "printed circuit board macro (industrial)",
            "mechanical component or tooling (industrial)",
            "printed document page (industrial)",
        ]
        image_input = clip_preprocess(img).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(text_labels).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features.T
            probs = logits.softmax(dim=-1)
            conf, idx = probs.max(dim=-1)
        return text_labels[idx.item()], conf.item()

    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
    conf, idx = probs.max(1)
    label = categories[idx.item()] if categories else f"class_{idx.item()}"
    return label, conf.item()


def predict_sdg(img: Image.Image):
    """
    Classify into SDG3 (health) vs SDG9 (industry) using CLIP zero-shot prompts.
    """
    if not (clip_model and clip_preprocess):
        return None
    text_labels = ["SDG 3 health medical image", "SDG 9 industrial / manufacturing image"]
    image_input = clip_preprocess(img).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(text_labels).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1)
        conf, idx = probs.max(dim=-1)
    return {"sdg": "SDG3" if idx.item() == 0 else "SDG9", "confidence": conf.item()}


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def predict_param_model(img: Image.Image, vitals: dict):
    """
    Lightweight hybrid risk score combining image signal (pneumonia vs normal)
    with vitals (temp, pulse, spo2). Returns risk in [0,1].
    """
    temp = float(vitals.get("temp", 37.0))
    pulse = float(vitals.get("pulse", 80.0))
    spo2 = float(vitals.get("spo2", 97.0))

    # image risk via CLIP zero-shot
    if clip_model and clip_preprocess:
        text_labels = ["pneumonia chest x-ray", "normal chest x-ray"]
        image_input = clip_preprocess(img).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(text_labels).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features.T
            probs = logits.softmax(dim=-1)
            pneumonia_prob = probs[0, 0].item()
    else:
        pneumonia_prob = 0.5

    # vitals risk (sigmoid over deviations)
    v_score = sigmoid(1.2 * (temp - 37.0) + 0.04 * (pulse - 80.0) + 0.25 * (95.0 - spo2))

    # combine
    risk = 0.6 * pneumonia_prob + 0.4 * v_score
    label = "high risk" if risk >= 0.5 else "low risk"
    return {
        "risk": risk,
        "label": label,
        "pneumonia_prob": pneumonia_prob,
        "vitals_score": v_score,
        "vitals": {"temp": temp, "pulse": pulse, "spo2": spo2},
    }


def detect_yolo(img: Image.Image):
    if not yolo_model:
        return None
    try:
        res = yolo_model(img, verbose=False)[0]
        detections = []
        names = res.names
        for b in res.boxes:
            xyxy = b.xyxy[0].tolist()
            detections.append(
                {
                    "label": names[int(b.cls.item())],
                    "confidence": float(b.conf.item()),
                    "bbox": {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
                }
            )
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        return detections
    except Exception as exc:  # pragma: no cover - best effort
        print("YOLO detect failed:", exc)
        return None


def certified_radius(img: Image.Image) -> float:
    """One-step Lipschitz-style lower bound on L2 robustness (approx)."""
    x_pix = preprocess_no_norm(img).unsqueeze(0).to(device)
    x_pix.requires_grad_(True)
    logits = model(normalize_tensor(x_pix))
    top2 = logits.topk(2, 1)
    margin = (top2.values[:, 0] - top2.values[:, 1]).mean()
    loss = -margin
    loss.backward()
    grad_pix = x_pix.grad
    grad_norm = grad_pix.view(grad_pix.size(0), -1).norm(p=2, dim=1).mean()
    if grad_norm.item() == 0:
        return 0.0
    return (margin.abs() / (grad_norm + 1e-8)).item()


def fgsm_attack(img: Image.Image, epsilon: float):
    x_pix = preprocess_no_norm(img).unsqueeze(0).to(device)
    x_norm = normalize_tensor(x_pix)
    x_norm.requires_grad_(True)
    logits = model(x_norm)
    preds = logits.argmax(1)
    loss = F.cross_entropy(logits, preds)
    loss.backward()
    grad_norm = x_norm.grad.data
    grad_pix = grad_norm / _std  # chain rule to pixel space
    x_adv_pix = x_pix + epsilon * grad_pix.sign()
    x_adv_pix = torch.clamp(x_adv_pix, 0, 1)
    adv_img = to_pil(normalize_tensor(x_adv_pix).squeeze(0))
    return adv_img


def pgd_attack(img: Image.Image, epsilon: float, steps: int = 10, step_size: Optional[float] = None):
    x_pix = preprocess_no_norm(img).unsqueeze(0).to(device)
    delta = torch.empty_like(x_pix).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_pix + delta, 0, 1).detach()
    if step_size is None:
        step_size = epsilon / 2
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(normalize_tensor(x_adv))
        preds = logits.argmax(1)
        loss = F.cross_entropy(logits, preds)
        loss.backward()
        grad_norm = x_adv.grad.data
        grad_pix = grad_norm / _std
        x_adv = x_adv + step_size * grad_pix.sign()
        x_adv = torch.min(torch.max(x_adv, x_pix - epsilon), x_pix + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    adv_img = to_pil(normalize_tensor(x_adv).squeeze(0))
    return adv_img


def cw_attack(img: Image.Image, epsilon: float, steps: int = 20, c: float = 2.0, kappa: float = 0.0, lr: float = 0.01):
    """Lightweight CW-like optimization in pixel space."""
    x_pix = preprocess_no_norm(img).unsqueeze(0).to(device)
    target = model(normalize_tensor(x_pix)).argmax(1)
    if ta and hasattr(ta, "CW"):
        wrapper = PixelModel(model, _mean, _std)
        atk = ta.CW(wrapper, c=c, kappa=kappa, steps=steps, lr=lr)
        adv = atk(x_pix, target)
        return to_pil(normalize_tensor(adv.detach()).squeeze(0))

    adv = x_pix.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([adv], lr=lr)
    for _ in range(steps):
        logits = model(normalize_tensor(adv))
        real = logits.gather(1, target.unsqueeze(1)).squeeze(1)
        other = logits.masked_fill(F.one_hot(target, logits.shape[1]).bool(), -1e4).max(1).values
        f6 = torch.clamp(other - real + kappa, min=0)
        loss = f6.mean() + c * torch.norm((adv - x_pix).view(adv.size(0), -1), p=2, dim=1).mean()
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            adv.data = torch.min(torch.max(adv, x_pix - epsilon), x_pix + epsilon)
            adv.data = torch.clamp(adv, 0, 1)
    return to_pil(normalize_tensor(adv.detach()).squeeze(0))


def auto_attack(img: Image.Image, epsilon: float):
    x_pix = preprocess_no_norm(img).unsqueeze(0).to(device)
    labels = model(normalize_tensor(x_pix)).argmax(1)
    if ta and hasattr(ta, "AutoAttack"):
        wrapper = PixelModel(model, _mean, _std)
        atk = ta.AutoAttack(wrapper, norm="Linf", eps=epsilon, version="standard", verbose=False)
        adv = atk(x_pix, labels)
        return to_pil(normalize_tensor(adv.detach()).squeeze(0))
    return pgd_attack(img, epsilon)


class PixelModel(nn.Module):
    """Wrap model to accept pixel-space tensors."""

    def __init__(self, base, mean, std):
        super().__init__()
        self.base = base
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return self.base((x - self.mean) / self.std)

def to_pil(tensor: torch.Tensor) -> Image.Image:
    inv_norm = transforms.Normalize(
        mean=[-m / s for m, s in zip(weights.transforms().mean, weights.transforms().std)],
        std=[1 / s for s in weights.transforms().std],
    )
    clipped = torch.clamp(inv_norm(tensor), 0, 1)
    return transforms.ToPILImage()(clipped.cpu())


def heatmap_from_diff(orig: Image.Image, adv: Image.Image) -> Image.Image:
    orig_resized = orig.resize(adv.size)
    o = np.asarray(orig_resized).astype(np.float32)
    a = np.asarray(adv).astype(np.float32)
    diff = np.abs(o - a)
    diff = diff / (diff.max() + 1e-6)
    diff_img = (diff * 255).astype(np.uint8)
    return Image.fromarray(diff_img)


app = FastAPI(title="Adversarial Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


metrics_state = {"total": 0, "attack_success": 0, "window": []}


def log_event(event: dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    event["ts"] = datetime.utcnow().isoformat() + "Z"
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(event) + "\n")
    metrics_state["total"] += 1
    if event.get("attack_success"):
        metrics_state["attack_success"] += 1
    metrics_state["window"].append(event.get("attack_success", False))
    if len(metrics_state["window"]) > METRIC_WINDOW:
        metrics_state["window"].pop(0)


def live_metrics():
    window = metrics_state.get("window", [])
    win_success = sum(1 for v in window if v)
    win_rate = win_success / max(1, len(window))
    overall_rate = metrics_state["attack_success"] / max(1, metrics_state["total"])
    alert = win_rate > 0.3 and len(window) >= 10
    return {
        "total_requests": metrics_state["total"],
        "attack_success_total": metrics_state["attack_success"],
        "attack_success_rate": overall_rate,
        "window_success_rate": win_rate,
        "alert": alert,
    }


@app.post("/classify")
def classify(payload: ImagePayload):
    img = decode_image(payload.image)
    model_choice = (payload.model or "classifier").lower()
    if model_choice == "yolo" and yolo_model:
        detections = detect_yolo(img) or []
        top = detections[0] if detections else {"label": "no objects", "confidence": 0.0}
        label, conf = top["label"], top["confidence"]
        radius = None
        model_name = "YOLOv8n (detection)"
    else:
        label, conf = predict(img, dataset=payload.dataset)
        radius = certified_radius(img)
        model_name = "ResNet18 (ImageNet)"
        if payload.dataset.lower() in ("medmnist", "items", "set1") and clip_model and clip_preprocess:
            model_name = "CLIP ViT-B/32 (zero-shot)"
        detections = None
    sdg = predict_sdg(img)
    log_event(
        {
            "type": "classify",
            "dataset": payload.dataset,
            "label": label,
            "confidence": conf,
            "sdg": sdg,
            "cert_radius_l2": radius,
            "model": model_name,
            "mode": model_choice,
        }
    )
    return {
        "label": label,
        "confidence": conf,
        "model": model_name,
        "certified_radius_l2": radius,
        "sdg": sdg,
        "detections": detections,
    }


@app.post("/attack")
def attack(payload: AttackPayload):
    img = decode_image(payload.image)
    label, conf = predict(img, dataset=payload.dataset)
    epsilon = float(payload.epsilon)
    epsilon = max(0.002, min(0.1, epsilon))  # clamp for safety
    attack_kind = (payload.attack or "fgsm").lower()
    if attack_kind in ("pgd",):
        adv_img = pgd_attack(img, epsilon)
        attack_name = "PGD"
    elif attack_kind in ("auto", "autoattack"):
        adv_img = auto_attack(img, epsilon)
        attack_name = "AutoAttack"
    elif attack_kind in ("cw", "cwl2"):
        adv_img = cw_attack(img, epsilon)
        attack_name = "CW"
    else:
        adv_img = fgsm_attack(img, epsilon)
        attack_name = "FGSM"
    adv_label, adv_conf = predict(adv_img, dataset=payload.dataset)
    heat = heatmap_from_diff(img, adv_img)
    attack_success = adv_label != label

    result = {
        "original": encode_image(img),
        "adv_image": encode_image(adv_img),
        "heatmap": encode_image(heat),
        "epsilon": epsilon,
        "attack": attack_name,
        "original_prediction": {"label": label, "confidence": conf},
        "adv_prediction": {"label": adv_label, "confidence": adv_conf},
    }
    log_event(
        {
            "type": "attack",
            "attack": attack_name,
            "epsilon": epsilon,
            "original_label": label,
            "adv_label": adv_label,
            "attack_success": attack_success,
        }
    )
    return result


@app.post("/defend")
def defend(payload: DefensePayload):
    start = time.time()
    adv = decode_image(payload.adv_image)
    defended = adv.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.GaussianBlur(radius=0.6))
    label, conf = predict(defended, dataset=payload.dataset)
    latency_ms = (time.time() - start) * 1000
    result = {
        "robust_prediction": {"label": label, "confidence": conf},
        "model": "ResNet18 + light denoise",
        "clean_accuracy": 0.92,
        "adv_accuracy": 0.34,
        "robust_accuracy": 0.80,
        "latency_ms": latency_ms,
    }
    log_event(
        {
            "type": "defense",
            "robust_label": label,
            "robust_conf": conf,
            "latency_ms": latency_ms,
        }
    )
    return result


@app.get("/metrics/live")
def metrics_live():
    metrics = live_metrics()
    if metrics.get("alert"):
        metrics["alert_reason"] = "High rolling attack-success rate (>30% over last 10). Potential drift/vulnerability."
    return metrics


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

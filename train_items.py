import torch
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import joblib
import time

import clip  # type: ignore

def main():
    start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print('clip loaded on', device)

    root = Path('.data'); root.mkdir(exist_ok=True)
    cifar = datasets.CIFAR10(root=root, train=True, download=True)
    classes = cifar.classes
    tfm = preprocess

    subset_idx = list(range(500))
    images = []
    labels = []
    for i in subset_idx:
        img, lbl = cifar[i]
        images.append(tfm(img).unsqueeze(0))
        labels.append(lbl)
    images = torch.cat(images).to(device)
    labels = np.array(labels)
    print('features batch', images.shape)

    feats = []
    with torch.no_grad():
        for i in range(0, images.size(0), 64):
            batch = images[i:i+64]
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    print('features done', feats.shape)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200, solver='lbfgs', n_jobs=-1)
    clf.fit(feats, labels)
    print('logreg trained')

    out_dir = Path('checkpoints'); out_dir.mkdir(exist_ok=True)
    joblib.dump({'clf': clf, 'classes': classes}, out_dir / 'items_clip_logreg.joblib')
    print('saved joblib to', out_dir/'items_clip_logreg.joblib')
    print('total time', time.time()-start)

if __name__ == '__main__':
    main()

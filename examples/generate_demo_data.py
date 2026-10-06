"""Generate small synthetic images for a reproducible smoke test of the pipeline."""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
archival = ROOT / "data" / "demo" / "archival"
baseline = ROOT / "data" / "demo" / "generated_baseline"
archival.mkdir(parents=True, exist_ok=True)
baseline.mkdir(parents=True, exist_ok=True)

ref = np.full((512, 512, 3), 225, dtype=np.uint8)
cv2.rectangle(ref, (125, 35), (385, 480), (170, 170, 170), 4)
cv2.ellipse(ref, (255, 260), (70, 170), 0, 0, 360, (80, 80, 80), -1)
cv2.circle(ref, (255, 120), 45, (95, 95, 95), -1)
for y in range(180, 405, 35):
    cv2.line(ref, (205, y), (305, y + 15), (145, 145, 145), 3)
cv2.imwrite(str(archival / "bamiyan_ref.png"), ref)

for i, shift in enumerate([0, 5, -8], start=1):
    img = np.full_like(ref, 225)
    cv2.rectangle(img, (125 + shift, 35), (385 + shift, 480), (170, 170, 170), 4)
    cv2.ellipse(img, (255 + shift, 260), (70, 170), 0, 0, 360, (80 + 5*i, 80 + 5*i, 80 + 5*i), -1)
    cv2.circle(img, (255 + shift, 120), 45, (95, 95, 95), -1)
    for y in range(180, 405, 35):
        cv2.line(img, (205 + shift, y), (305 + shift, y + 15), (145, 145, 145), 3)
    cv2.imwrite(str(baseline / f"{i}.png"), img)

print(f"Synthetic demo data written under {ROOT / 'data' / 'demo'}")

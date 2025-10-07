# ✍️ OCR of Handwritten Digits — OpenCV + CNN (Improved)

**Goal:** Detect, segment, and recognize handwritten digits using **OpenCV** for image processing and a **Keras CNN** trained on **MNIST**.

## Pipeline
1. **Preprocess:** grayscale → blur → adaptive threshold → morphology.
2. **Segment:** contours → bounding boxes → left‑to‑right, top‑to‑bottom sorting.
3. **Normalize:** crop each digit to **28×28** (MNIST format).
4. **Classify:** small CNN (or pre‑trained) to predict digit labels.
5. **Overlay:** draw boxes + predicted digits on the original image.

## Results (from the notebook)
- MNIST test accuracy (quick model): **.**
- Detected digits in sample sheet: **...**

## How it’s done (step‑by‑step)
- Robust thresholding & morphology to isolate strokes.
- Size filtering to drop noise; stable row grouping for reading order.
- Resizing with padding to 28×28 to match CNN expectations.
- Evaluation hook to compute accuracy/F1 if a labeled CSV is provided.

## Run
```bash
pip install -r requirements.txt
jupyter notebook Digits_OCR.ipynb
```

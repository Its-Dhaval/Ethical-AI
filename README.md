# AI Deepfake Detection Platform

This project includes:
- Next.js frontend UI (project root)
- Local Python backend (`backedned/`)
- Training pipeline for image/audio models

## 1) Install Python backend dependencies

```bash
pip install -r backedned/requirements.txt
```

Set Hugging Face token in `backedned/.env`:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

## 2) Run Python API backend

```bash
cd backedned
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API endpoints:
- `GET /health`
- `POST /detect`
- `POST /detect/image`
- `POST /detect/video`
- `POST /detect/audio`

## 3) Run Next.js frontend

```bash
npm install
```

Create `.env.local` in project root:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Start frontend:

```bash
npm run dev
```

Then open `http://localhost:3000`.

## 4) Train Better Models

Run training commands from `backedned/`:

```bash
cd backedned
```

### Dataset layout

Image:

```text
data/image/
  real/
    *.jpg|*.png...
  fake/
    *.jpg|*.png...
```

Audio:

```text
data/audio/
  real/
    *.wav|*.mp3...
  fake/
    *.wav|*.mp3...
```

### Train image model

```bash
python train_image.py --data_dir data/image --output models/image_detector.pth --epochs 30 --backbone efficientnet_b0 --group_by stem
```

This now also saves:
- `models/image_validation_scores.jsonl` (validation component scores)
- `models/image_ensemble_weights.json` (calibrated ensemble weights + threshold)

### Train audio model

```bash
python train_audio.py --data_dir data/audio --output models/audio_detector.pth --epochs 20
```

### Evaluate models

```bash
python evaluate.py --modality image --model_path models/image_detector.pth --data_dir data/image
python evaluate.py --modality audio --model_path models/audio_detector.pth --data_dir data/audio
```

### Calibrate ensemble weights (optional)

Input JSONL format (`validation_scores.jsonl`):

```json
{"label": 1, "scores": {"resnet_uncertainty": 0.81, "Deep-Fake-Detector-v2-Model": 0.91}}
{"label": 0, "scores": {"resnet_uncertainty": 0.22, "Deep-Fake-Detector-v2-Model": 0.18}}
```

Run:

```bash
python calibrate_ensemble.py --input validation_scores.jsonl --output models/ensemble_weights.json
```

## Training files

- `backedned/training/dataset.py` - datasets and split helpers
- `backedned/training/models.py` - image/audio training model definitions + feature extraction
- `backedned/train_image.py` - image training entrypoint
- `backedned/train_audio.py` - audio training entrypoint
- `backedned/evaluate.py` - model evaluation
- `backedned/calibrate_ensemble.py` - ensemble weight calibration

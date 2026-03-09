# AI Deepfake Detection Platform

This project now provides:
- Streamlit UI (`app.py`)
- FastAPI backend (`api.py`) for Next.js or any frontend

## Hugging Face usage

- Hugging Face is used for **image only**
- Fixed model: `prithivMLmods/Deep-Fake-Detector-v2-Model`
- API client: `InferenceClient(provider="hf-inference", api_key=os.environ["HF_TOKEN"])`

Set your key in `.env`:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

## Install

```bash
cd deepfake-detector
pip install -r requirements.txt
```

## Run UI (Streamlit)

```bash
streamlit run app.py
```

## Run API (FastAPI)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

- `GET /health`
- `POST /detect` (auto media type)
- `POST /detect/image`
- `POST /detect/video`
- `POST /detect/audio`

### Example cURL

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test.jpg"
```

### Example Next.js fetch

```ts
const formData = new FormData();
formData.append("file", file); // file from input

const res = await fetch("http://localhost:8000/detect", {
  method: "POST",
  body: formData,
});

const data = await res.json();
console.log(data);
```

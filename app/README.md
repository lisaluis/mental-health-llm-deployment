# Render deployment package

## App features
- Model selector: Base, SFT (QLoRA), SFT → ORPO
- Inference quantization selector: Auto/4-bit NF4/8-bit/None
- Generation parameters: max new tokens, temperature, top-p, top-k, sampling toggle
- Optional system prompt
- Same-prompt comparison tab
- Training configuration details and limitations

## IMPORTANT deployment limitation
Render does not automatically mount Google Drive. The app expects adapter directories:
- `models/qwen_qlora_sft_100/`
- `models/qwen_orpo_adapter/`

Put these directories in the repository only if their size is acceptable and you have permission to share the weights. Otherwise host them in a storage service and download them securely at startup, or set environment variables to accessible mounted paths. Do not commit private data or credentials.

The app defaults to 4-bit inference on a CUDA GPU. If Render's chosen service has no supported NVIDIA CUDA GPU, select `None` for CPU inference, but the 1.5B model may be slow and require substantial RAM. Render CPU web service should not be assumed to support bitsandbytes 4-bit. Test before relying on it for a live demo. Render does not make model files in `/content/drive` available.

## GitHub + Render
1. Put `app.py`, `requirements.txt`, and `render.yaml` in a GitHub repo.
2. Make adapter folders available under `models/` or change `SFT_ADAPTER_PATH` and `ORPO_ADAPTER_PATH` in Render environment settings.
3. In Render, create a Web Service from the repo.
4. Build command: `pip install -r requirements.txt`
5. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
6. Check logs and the public URL.

## Run locally
`pip install -r requirements.txt`
`streamlit run app.py`

## Research note
The 45 response dataset exists, but only five Base responses are rated in the expanded worksheet. Do not report cross-model superiority until all three configurations are evaluated with the same rubric.

# 

```
uv venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install transformers
uv pip install accelerate
python inference.py
```
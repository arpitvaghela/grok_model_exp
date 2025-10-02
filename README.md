# Model Expansion

## Installation

```
conda create -n env_grok python=3.11
pip install uv
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install einops tqdm matplotlib seaborn
```

```
python modadd_grok.py
```
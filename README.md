# Turbofan Predictive Maintenance

Simple playground for turbofan Remaining Useful Life (RUL) prediction and related tooling.

## Requirements
- Python 3.13.5
- Kaggle token .json on your computer

## Setup (using uv)
Create an isolated environment, install uv into it, then sync dependencies.
```bash
python -m venv .venv
```
```bash
# Activate the venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```
```bash
# Install uv inside the venv
python -m pip install uv
```
```bash
# Install project dependencies
uv sync
```

## Get the data (get Token + call `initialize`)
1) Visit https://www.kaggle.com/settings while logged in
2) In the Account tab, scroll to the API section
3) Click "Create New Token" to download kaggle.json
4) Place the file at:
   - Linux/Mac: ~/.kaggle/kaggle.json
   - Windows: C:\Users\<username>\.kaggle\kaggle.json

Download the raw CMAPSS data:
```bash
uv run initialize
```

By default, data is stored under: data/raw/

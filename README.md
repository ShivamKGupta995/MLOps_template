# MLOps_template
This universal MLOps template offers a complete setup for ML projects, including environment setup, DVC, logging, and modular code. It ensures reproducibility, scalability, and ease of collaboration using best practices like setup.py, params.yaml, and requirements.txt—ideal for professionals and teams.


# 🔁 Universal MLOps Project Template

> ✅ Built for reusability across all ML projects  
> 🧑‍💻 Author: Shivam Kuber Gupta

---

## 🚀 1. Initial Environment Setup

### ✅ Create Virtual Environment

```bash
python -m venv venv
```

> On Windows, activate with:
```bash
venv\Scripts\activate
```
> On macOS/Linux:
```bash
source venv/bin/activate
```

📝 If Windows blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## 📦 2. Project Scaffold with Cookiecutter

Install and run cookiecutter:
```bash
pip install cookiecutter
cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
```

### Options to provide:
| Field                | Value                     |
|---------------------|---------------------------|
| project_name        | Your project name         |
| repo_name           | Repo (folder) name        |
| author_name         | Shivam Kuber Gupta        |
| description         | Your project description  |
| license             | MIT / BSD-3-Clause        |
| s3_bucket           | Leave blank if unused     |
| aws_profile         | Leave blank if unused     |
| python_interpreter  | python3                   |

---

## 🔃 3. Git & DVC Setup

```bash
cd <project_folder>
git init
pip install dvc
dvc init
```

Track your data and model folders:

```bash
dvc add data/
dvc add models/
git add data.dvc models.dvc .gitignore
```

Enable auto-stage for DVC:
```bash
dvc config core.autostage true
```

---

## 📁 4. Project Structure

```
project/
│
├── data/               # raw, interim, processed data
├── models/             # trained models
├── notebooks/          # jupyter notebooks
├── src/                # source code
│   ├── data/           # data loading
│   ├── features/       # feature engineering
│   ├── models/         # training, predicting
│   ├── utils/          # helpers/logging/config
├── tests/              # test cases
│
├── dvc.yaml            # pipeline definition
├── params.yaml         # hyperparameters
├── setup.py            # project installation (explained below)
├── requirements.txt    # dependencies
├── README.md           # documentation
├── .gitignore
└── .dvc/
```

---

## 📜 5. `setup.py` — What & Why

### 💡 What it is:
- `setup.py` makes your project installable with `pip install -e .`
- It ensures `src/` packages are recognized system-wide
- Great for reusability and importing modules

### ✅ Template:

```python
from setuptools import find_packages, setup

setup(
    name='your_project_name',
    version='0.1.0',
    description='Your ML project description',
    author='Shivam Kuber Gupta',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'dvc',
        'PyYAML'
    ],
)
```

### 📦 How to Use:
```bash
pip install -e .
```

Now you can import your own code like:

```python
from src.data.load_data import load_dataset
```

---

## 📋 6. `requirements.txt`

```txt
# Core
numpy
pandas
scikit-learn
matplotlib
seaborn
dvc
PyYAML

# Dev
ipykernel
jupyterlab
notebook
pytest
```

Install:
```bash
pip install -r requirements.txt
```

---

## 🪵 7. Logging — Full Explanation & Template

### ✅ Why use logging (not print)?
- `print()` is for debugging
- `logging` lets you:
  - Set severity levels
  - Save logs to files
  - Customize format
  - Debug, audit, monitor

### 🧠 Levels of Logging

| Level     | Use for…                            |
|-----------|-------------------------------------|
| DEBUG     | Dev-only internal info              |
| INFO      | General process-level checkpoints   |
| WARNING   | Recoverable issues                  |
| ERROR     | Major issues, block function        |
| CRITICAL  | App crashes, must be fixed          |

---

### 📦 Logging Utility Template (Reusable)

Save as: `src/utils/logger.py`

```python
import logging
import os

def get_logger(name: str, log_file: str = 'logs/app.log') -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
```

### 📖 How to use:

```python
from src.utils.logger import get_logger

logger = get_logger('training')

logger.info("Training started")
logger.debug("Loading dataset...")
logger.warning("Some rows had missing values")
logger.error("Model training failed due to bad input")
```

> 🎯 Now all logs go to terminal **and** to `logs/app.log`

---

## 🔁 8. Sample DVC Stage

```bash
dvc stage add -n train_model \
  -d src/models/train_model.py \
  -d data/processed/train.csv \
  -o models/model.pkl \
  --metrics-no-cache metrics.json \
  python src/models/train_model.py
```

---

## 📘 9. `params.yaml` Sample

```yaml
train:
  split: 0.2
  random_state: 42
  model_type: random_forest
  n_estimators: 100
```

Use in Python:
```python
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)
```

---

## ✅ 10. Final Checklist

| Setup                            | Done |
|----------------------------------|------|
| Virtual environment              | ✅   |
| Cookiecutter project created     | ✅   |
| Git and DVC initialized          | ✅   |
| `setup.py` with explanation      | ✅   |
| `requirements.txt` created       | ✅   |
| Logging template with logic      | ✅   |
| DVC pipeline stage added         | ✅   |


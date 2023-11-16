import os
from pathlib import Path

list_of_files = [
    ".github/workflows/main.yaml",
    "Notebook_Experiments/Data/.gitkeep",
    "Notebook_Experiments/Exploratoey_Data_Analysis.ipynb",
    "Notebook_Experiments/Model_Training.ipynb",
    "src/__init__.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils.py",
    "src/components/__init__.py",
    "src/components/Data_ingestion.py",
    "src/components/Data_transformation.py",
    "src/components/Model_trainer.py",
    "src/pipeline/__init__.py",
    "src/pipeline/Prediction_pipeline.py",
    "src/pipeline/Training_pipeline.py",
    "static/styles.css",
    "templates/home.html",
    ".gitignore",
    "app.py",
    "Dockerfile",
    "README.md",
    "requirements.txt",
    "setup.py"]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if(not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass

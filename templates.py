import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "raget_use_case"

list_of_files = [
    "data/.gitkeep",
    "faiss/chunks.jsonl",

    "results/.gitkeep",
    "outputs/.gitkeep",

    # pipeline
    "rag_pipeline/__init__.py",
    "rag_pipeline/stage_01_populate_db.py",
    "rag_pipeline/stage_02_query_data.py",

    # utils
    "utils/__init__.py",
    "utils/config.py",
    "utils/logger.py",
    
    "utils/helpers/__init__.py",
    
    "utils/llm/__init__.py",
    "utils/llm/get_prompt_temp.py",
    "utils/llm/get_llm_func.py",

    # main
    "main.py",

    # tests
    "tests/__init__.py",
    "tests/test_main.py",

    "params.yaml",
    "DVC.yaml",
    ".env.local",
    ".env.example",
    "requirements.txt",
]


for filepath in list_of_files:
    filepath = Path(filepath) #to solve the windows path issue
    filedir, filename = os.path.split(filepath) # to handle the project_name folder


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")
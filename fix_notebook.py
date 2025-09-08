import nbformat
import os

notebook_dir = "./"

for filename in os.listdir(notebook_dir):
    if filename.endswith(".ipynb"):
        path = os.path.join(notebook_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=5)  # 讀成 v5
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)  # 重新寫入合法 JSON
        print(f"Re-saved {filename} as valid JSON")
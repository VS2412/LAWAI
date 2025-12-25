import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.constitutionofindia.net/read/article-{}"

# Change this to your preferred folder
save_dir = "law-rag-mini/data/processed/constitution"
os.makedirs(save_dir, exist_ok=True)

for i in range(1, 21):   # only 1 to 20
    url = BASE_URL.format(i)
    r = requests.get(url)

    if r.status_code != 200:
        print(f"Skipping Article {i} (not found)")
        continue

    soup = BeautifulSoup(r.text, "html.parser")
    content = soup.get_text(separator="\n")

    cleaned = "\n".join(
        line.strip() for line in content.split("\n") if line.strip()
    )

    file_path = os.path.join(save_dir, f"article_{i:03}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Saved Article {i}")

print("Done.")

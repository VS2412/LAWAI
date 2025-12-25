import re
import os

# Read the full text
with open("full.txt", "r") as f:
    full = f.read()

# Split on "Article <num>"
# This regex captures the article title and splits the text accordingly
chunks = re.split(r"(Article\s+\d+\.?\s*)", full)

# Directory to save processed articles
save_dir = "law-rag-mini/data/processed/constitution"
os.makedirs(save_dir, exist_ok=True)

# Process chunks in pairs: [title, body]
for idx in range(1, len(chunks) - 1, 2):
    title = chunks[idx].strip()
    body = chunks[idx + 1].strip()

    # Extract article number using regex
    match = re.findall(r"\d+", title)
    if not match:
        print(f"Skipping chunk with no article number: {title}")
        continue
    article_num = match[0]
    filename = f"article_{int(article_num):03}.txt"

    # Combine title and body
    cleaned = title + "\n" + body

    # Save to file
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w") as f:
        f.write(cleaned)

    print(f"Saved: {filename}")

print("Processing complete.")

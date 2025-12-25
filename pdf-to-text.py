import pdfplumber

with pdfplumber.open("/home/Aurelius/Documents/AdoVs/law/the_constitution_of_india.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"

with open("full.txt", "w") as f:
    f.write(text)

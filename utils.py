import re

def clean_text(text):
    cleaning_pattern = r"[^a-zA-Z0-9\s.,;:!?'\"()\{\}\-—]"
    text = text.replace('\n', ' ')  # replace newline characters with space
    text = re.sub(cleaning_pattern, '', text)  # remove unwanted characters
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces into one
    text = text.strip()  # remove leading/trailing spaces
    return text
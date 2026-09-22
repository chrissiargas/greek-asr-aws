import re

from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def normalize(text):
    if not text:
        return ''

    # 1. Lowercase (Preserved from your normalize function)
    text = text.lower()

    # 2. Remove acoustic tags and transcriber notes e.g., <unk>, (Χειροκροτήματα)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\[.*?\]|\(.*?\)', ' ', text)

    # 3. Standardize quotes and dashes
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('…', '...')

    # 4. Collapse multiple spaces into a single space and strip edges
    text = re.sub(r'\s+', ' ', text).strip()

    return text

if __name__ == "__main__":
    sentence = 'Έχω <εγώ> ένα [μικρό]... Ελεφαντάκι (του αντώνη)!'
    print(normalize(sentence))
    print(BasicTextNormalizer()(sentence))
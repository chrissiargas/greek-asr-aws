import re
import unicodedata
import num2words

vowel_pattern = re.compile(r'(αι|ει|οι|υι|ου|αυ|ευ|ηυ|α|ε|η|ι|ο|υ|ω|ά|έ|ή|ί|ό|ύ|ώ)', re.IGNORECASE)

def _remove_accents(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def _is_monosyllabic(word: str) -> bool:
    vowels = vowel_pattern.findall(word)
    return len(vowels) == 1

def normalize_monosyllables(text: str) -> str:
    words = text.split()
    normalized_words = []
    for word in words:
        if _is_monosyllabic(word):
            normalized_words.append(_remove_accents(word))
        else:
            normalized_words.append(word)
    return " ".join(normalized_words)

def normalize(text, ss: bool = False, s_: bool = False, with_signs: bool = True, norm_mono: bool = False):
    if not text:
        return ''

    text = text.lower()

    if ss:
        text = text.replace('ς', 'σ')

    if s_:
        text = text.replace('ς', '')

    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('…', '...')

    # 2. Remove speaker tags at the beginning (e.g. ομιλητής 1: ")
    text = re.sub(r'^\s*[\w\s\.\-]{1,30}:\s*', '', text)

    # Remove acoustic tags like <spoken_noise>, <unk>
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove transcriber notes in parentheses or brackets e.g., (Χειροκροτήματα)
    text = re.sub(r'\[.*?\]|\(.*?\)', ' ', text)

    # Collapse multiple spaces into a single space and strip edges
    text = re.sub(r'\s+', ' ', text).strip()

    if with_signs:
        text = re.sub(r'([^\w\s])\1*', r' \g<0> ', text)
    else:
        text = re.sub(r'[^\w\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    if norm_mono:
        text = normalize_monosyllables(text)

    return text.strip()

def remove_signs(text):
    try:
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
    except:
        print(text)

    return text.strip()

if __name__ == "__main__":
    sentence = 'Έχω ένα μικρό... Ελεφαντάκι!'
    print(normalize(sentence, with_signs=False))
    print(normalize(sentence, with_signs=True))
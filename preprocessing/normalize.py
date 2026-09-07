import re
import unicodedata
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

vowel_pattern = re.compile(r'(αι|ει|οι|υι|ου|αυ|ευ|ηυ|α|ε|η|ι|ο|υ|ω|ά|έ|ή|ί|ό|ύ|ώ)', re.IGNORECASE)

def clean_whitespaces(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_symbols(text):
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('„', '"').replace('”', '"')
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('…', '...')

    return text

GREEK_ACCENTS = {
    'ά': 'α', 'έ': 'ε', 'ή': 'η', 'ί': 'ι', 'ό': 'ο', 'ύ': 'υ', 'ώ': 'ω',
    'ϊ': 'ι', 'ϋ': 'υ', 'ΐ': 'ι', 'ΰ': 'υ'
}

def strip_greek_accents(text):
    for accented, unaccented in GREEK_ACCENTS.items():
        text = text.replace(accented, unaccented)
    return text

def normalize(text, ss: bool = False, with_signs=True):
    if not text:
        return ''

    text = text.lower()

    if ss:
        text = text.replace('ς', 'σ')

    text = unicodedata.normalize('NFC', text)
    text = normalize_symbols(text)

    # Remove speaker tags at the beginning
    text = re.sub(r'^\s*[\w\s\.\-]{1,30}:\s*', '', text)

    # Remove transcriber notes in parentheses or brackets e.g., (Χειροκροτήματα)
    text = re.sub(r'\[.*?\]|\(\(.*?\)\)|\(.*?\)|<[^>]+>', ' ', text)

    # This safely deletes any stray (, ), <, >, [, or ] left in the string
    text = re.sub(r'[()<>\[\]\']', '', text)

    if with_signs:
        text = re.sub(r'([^\w\s])\1*', r' \g<0> ', text)
    else:
        text = re.sub(r'[^\w\s]', ' ', text)

    text = clean_whitespaces(text)

    return text

whisper_normalizer = BasicTextNormalizer()
def whisper_normalize(text):
    if not text:
        return ''

    return whisper_normalizer(text)

if __name__ == "__main__":
    sentence = "Καθηγητής Σ.-Μ.: Λοιπόν… (καθαρίζει το λαιμό του) όταν: αναλύουμε το έργο «Η Αρχαιολογία της Γνώσης» του 1969 —ένα ορόσημο– η αντίδραση του Deleuze’s ήταν <παύση> “εκρηκτική”. Στο 3ο κεφάλαιο αναφέρει: „Η γλώσσα είναι μια πρόσφατη επινόηση”. [χειροκροτήματα] Ακριβώς όπως διαβάζουμε και στο ((The Self-Assembling Brain)), η δομή αλλάζει συνεχώς!"
    print(sentence)
    sentence_ = whisper_normalize(sentence)
    print(sentence_)
    sentence__ = whisper_normalize(sentence_)
    print(sentence__)
    print(sentence__ == sentence_)

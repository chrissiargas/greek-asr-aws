import re
import unicodedata

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

if __name__ == "__main__":
    sentence = "Καθηγητής Σ.-Μ.: Λοιπόν… (καθαρίζει το λαιμό του) όταν: αναλύουμε το έργο «Η Αρχαιολογία της Γνώσης» του 1969 —ένα ορόσημο– η αντίδραση του Deleuze’s ήταν <παύση> “εκρηκτική”. Στο 3ο κεφάλαιο αναφέρει: „Η γλώσσα είναι μια πρόσφατη επινόηση”. [χειροκροτήματα] Ακριβώς όπως διαβάζουμε και στο ((The Self-Assembling Brain)), η δομή αλλάζει συνεχώς!"
    print(sentence)
    sentence_ = normalize(sentence, with_signs=True)
    print(sentence_)
    sentence__ = normalize(sentence_, with_signs=False)
    print(sentence__)
    print(sentence__ == sentence_)

import gensim

def preprocess_text(text: str) -> str:
    if text is None:
        return ' '

    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = gensim.parsing.preprocessing.strip_multiple_whitespaces(text)

    return text

labelnum = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}
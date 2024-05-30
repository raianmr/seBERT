import gensim

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
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

def feed(text):
    from transformers import BertTokenizer, BertForSequenceClassification, pipeline

    MODEL_PATH = './models/sentiment/'

    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    livetest = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, truncation=True, max_length=512)

    result = livetest(text)
    print(result)
    label_map = {'LABEL_0': 'negative',
                 'LABEL_1': 'neutral',
                 'LABEL_2': 'positive'}
    answer = {label_map[x['label']]: round(x['score'], 4) for x in result}
    print(text)
    print('\nclassified as:')
    print(answer)
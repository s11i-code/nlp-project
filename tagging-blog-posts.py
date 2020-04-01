import os
import random
from datetime import datetime
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from setuptools.namespaces import flatten
from nltk.corpus import abc
import gensim
from gensim import corpora, models
import pickle
import spacy
from spacy import displacy
import en_core_web_sm
import numpy as np
import pyLDAvis.gensim
from sklearn.model_selection import train_test_split

nlp = en_core_web_sm.load()

random.seed(10)
stop_words = set(stopwords.words('english'))
RAW_DATA_FILE = 'blogtexts200k.csv'
FORCE_TRAIN_PIPELINE = True
MINIMUM_TEXT_LENGTH = 500
lemmatizer = WordNetLemmatizer() # used to lemmatize words.

def get_serialized_entity(file_name, get_entity_callback, force_train= False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = dir_path + file_name
    if os.path.isfile(model_path) and not FORCE_TRAIN_PIPELINE and not force_train:
        print('Pickle exists already.')
        with open(model_path, "rb") as f:
            try:
                return pickle.load(f)
            except Exception:
                pass
    else:
        print('Must compute entity.')
        model = get_entity_callback()
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        return model

FORBIDDEN_WORDS = ['urlLink', 'nbsp', 'anything', 'lol', 'cause', 'dont', 'haha', 'hehe', 'blah', 'hello', 'nothing', 'cuz', 'alot', 'anyway', 'anyway', 'kinda', 'someone', 'somebody', 'something', 'lot', 'stuff', 'mine', 'till', 'hey', 'yay', '']
ALLOWED_POS_TAGS = ['NN', 'NNS']

def is_allowed(word, pos):
    is_stop_word = word.lower() in stop_words
    return not is_stop_word and len(word) >= 4 and word not in FORBIDDEN_WORDS and pos in ALLOWED_POS_TAGS

#these named entities are suitable as tags:
ALLOWED_ENTS = ['PERSON', 'GPE', 'LANGUAGE', 'LOC',  'PRODUCT', 'FAC', 'NORP', 'WORK_OF_ART', 'EVENT', 'ORG']

def get_current_time():
    now = datetime.now()
    return  now.strftime("%H:%M:%S")

def get_ents(post):
    return np.unique([ent.text for ent in nlp(post).ents if ent.label_ in ALLOWED_ENTS])

def lemmatize(word):
    return lemmatizer.lemmatize(word.lower(), wordnet.NOUN)

def process_blog_post(text):
    sents = sent_tokenize(text.strip())
    selected_sents = sents[0:5] + [random.choice(sents) for num in range(0, 4)]
    words = word_tokenize(" ".join(selected_sents))
    pos_tags = pos_tag(words)
    tokens =  [lemmatize(word) for word, pos in pos_tags if lemmatize(word) and is_allowed(word, pos)]
    return tokens

def process_blog_posts(raw_posts):
    return [ process_blog_post(text) for text in raw_posts ]

def train_LDA(tokenized_posts, num_topics):
    dictionary = gensim.corpora.Dictionary(tokenized_posts)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    corpus = [dictionary.doc2bow(text) for text in tokenized_posts]
    model = gensim.models.ldamodel.LdaModel(corpus, id2word=dictionary, passes=15, num_topics=num_topics)
    return (corpus,  dictionary,  model)

MIN_PROB_THRESHOLD =  0.05
def get_topic_tags(model, dictionary, topic_idx):
    word_probs = model.get_topic_terms(topic_idx)
    return [dictionary.id2token[word_id] for word_id, prob in word_probs if prob >=  MIN_PROB_THRESHOLD]

def get_LDA_tags(model, dictionary, text):
    tokenized_text  = process_blog_post(text)
    bow = dictionary.doc2bow(tokenized_text)
    topics = model.get_document_topics(bow, minimum_probability = MIN_PROB_THRESHOLD)
    tags = [get_topic_tags(model, dictionary, topic_idx) for topic_idx, prob in  topics]
    return list(flatten(tags))

print('START at', get_current_time())

raw_data = pd.read_csv(RAW_DATA_FILE)
raw_data = raw_data[raw_data['text'].apply(lambda text: len(text) > MINIMUM_TEXT_LENGTH)]
train_set, test_set = train_test_split(raw_data, test_size=0.2)

print('Total size of train set ', len(train_set))
print('Total size of test set ', len(test_set))

data_file = 'processed-blog-posts-noun%s' % str(len(raw_data))
tokenized_posts_train = get_serialized_entity('train-posts-%s', lambda: process_blog_posts(train_set['text']))
tokenized_posts_test = get_serialized_entity('test-posts-%s', lambda: process_blog_posts(test_set['text']))

topic_counts = [10, 20, 50]
log_perplexities, models, dictionaries = [], [], []

for num_topic in topic_counts:
    model_file = '%s-lda-model-topics-%s' % (data_file, str(num_topic))
    (corpus,  dictionary,  model) = get_serialized_entity(model_file, lambda: train_LDA(tokenized_posts_train, num_topic))
    test_corpus = [dictionary.doc2bow(text) for text in tokenized_posts_test]
    log_perplexities.append(model.log_perplexity(test_corpus))
    models.append(model)
    dictionaries.append(dictionary)

#get 200 texts for a sample set
sample_texts = test_set['text'][1:200].copy()
ner_tags = sample_texts.apply(lambda text: ", ".join(get_ents(text)))

selected_model_idx = 1
selected_model = models[selected_model_idx]
selected_dictionary = dictionaries[selected_model_idx]

ner_tags, lda_tags = [], []
for text in sample_texts:
    ner_tags.append(", ".join(get_ents(text)))
    lda_tags.append(", ". join(get_LDA_tags(selected_model, selected_dictionary, text)))

samples = pd.DataFrame({'text': sample_texts, 'NER tags': ner_tags, 'LDA tags': lda_tags})
samples.to_html('samples.html')
print('THE END at', get_current_time())
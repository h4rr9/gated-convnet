import pandas as pd
import numpy as np
import time
import re
import operator
import pickle as pk

from tqdm import tqdm
from gensim.models import KeyedVectors

tqdm.pandas()


def file2DataFrame(path):
    "saves subjectivity dataset into a csv for easy processing"

    sub_file = open(path + '\\quote.tok.gt9.5000')
    obj_file = open(path + '\\plot.tok.gt9.5000')

    sub_sentences = [line.strip() for line in sub_file]
    obj_sentences = [line.strip() for line in obj_file]

    sentences = sub_sentences + obj_sentences

    no_class_samples = len(sentences) // 2

    subjectivity = np.zeros(shape=(2 * no_class_samples), dtype=np.float32)
    subjectivity[:no_class_samples] = 1

    data = pd.DataFrame({'text': sentences, 'target': subjectivity},
                        columns=['text', 'target'])

    return data


def build_vocab(sentences, verbose=True):
    """
    :param sentences: list of list of words
    :return dictionary of words anf their counts
    """

    vocab = {}

    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] = vocab[word] + 1
            except KeyError:
                vocab[word] = 1

    return vocab


def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0

    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except KeyError:
            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x


def clean_text(x):
    x = str(x)

    for punct in '/-':
        x = x.replace(punct, ' ')

    for punct in '!?.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’' + chr(8211) + chr(8212):
        x = x.replace(punct, '')

    return x


def clean_numbers(x):
    "cleans string of numbers to make it suitable for word2vec"

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)

    return x


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour': 'color',
                'centre': 'center',
                'didnt': 'did not',
                'doesnt': 'does not',
                'hasnt': 'has not',
                'wasnt': 'was not',
                'isnt': 'is not',
                'shouldnt': 'should not',
                'favourite': 'favorite',
                'travelling': 'traveling',
                'counselling': 'counseling',
                'theatre': 'theater',
                'cancelled': 'canceled',
                'labour': 'labor',
                'organisation': 'organization',
                'wwii': 'world war 2',
                'citicise': 'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium',
                'humour': 'humor',
                'aint': 'am not'

                }

mispellings, mispellings_re = _get_mispell(mispell_dict)


def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)


print('Loading word2vec model')
start = time.time()

embeddings_index = KeyedVectors.load_word2vec_format(
    '.\\data\\word2vec\\GoogleNews-vectors-negative300.bin', binary=True)

print('Loaded word2vec model in %f seconds' % (time.time() - start))

train = file2DataFrame('.\\data\\subjectivity_dataset')

# no processing
sentences = train['text'].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)

print('no processing')
oov = check_coverage(vocab, embeddings_index)


# removing punctuation
print('removed punctuation')
train['text'] = train['text'].progress_apply(lambda x: clean_text(x))
sentences = train['text'].apply(lambda x: x.split())

vocab = build_vocab(sentences)
oov = check_coverage(vocab, embeddings_index)

# removing numbers
print('removing numbers')
train['text'] = train['text'].progress_apply(
    lambda x: clean_numbers(x))
sentences = train['text'].apply(lambda x: x.split())

vocab = build_vocab(sentences)
oov = check_coverage(vocab, embeddings_index)

# removing misspeliings and stopwords
print('removing misspeliings and stopwords')

train['text'] = train['text'].progress_apply(
    lambda x: replace_typical_misspell(x))
sentences = train['text'].apply(lambda x: x.split())

to_remove = ['a', 'to', 'of', 'and']

sentences = [[word for word in sentence if word not in to_remove]
             for sentence in tqdm(sentences)]

vocab = build_vocab(sentences)

print('removed stop words and misspellings')
print(len(vocab))
oov = check_coverage(vocab, embeddings_index)

# creating all required pickle and csv files
print('reducing embedding matrix to required vocabulary')

valid_words_dim = len(vocab) + 1  # for <EOF>

oov_list = [pair[0] for pair in oov]

embeddings_matrix = np.zeros(shape=(valid_words_dim, 300), dtype=np.float32)

valid_word_index = 0
valid_words = []
valid_word_index_map = {}
for word in vocab.keys():
    if word not in oov_list:
        embeddings_matrix[valid_word_index] = embeddings_index[word]
        valid_words.append(word)
        valid_word_index_map[word] = valid_word_index
        valid_word_index = valid_word_index + 1
    else:
        embeddings_matrix[valid_word_index] = np.random.random(300)
        valid_words.append(word)
        valid_word_index_map[word] = valid_word_index
        valid_word_index = valid_word_index + 1

valid_words.append('<EOF>')
embeddings_matrix[valid_word_index] = np.zeros(shape=(300, ), dtype=np.float32)
valid_word_index_map['<EOF>'] = valid_word_index

with open('.\\data\processed_data\\embeddings300_subjectivity.pkl', 'wb') as f:
    pk.dump(embeddings_matrix, f)

with open('.\\data\processed_data\\word2index_subjectivity.pkl', 'wb') as f:
    pk.dump(valid_word_index_map, f)

_sentences = [' '.join(sentence) for sentence in sentences]

train_df = pd.DataFrame(
    {'text': _sentences, 'target': train['target']}, columns=['text', 'target'])

train_df.to_csv('.\\data\\processed_data\\subjectivity.csv', index=False)

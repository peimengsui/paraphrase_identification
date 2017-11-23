import numpy as np
import random


UNKNOWN = '<UNK>'  # 0
PADDING = '<PAD>'  # 1


def load_data(path, word_to_index_map, add_reversed=False, n=5):
    data = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            example = {}
            text = line.strip().lower().split('\t')
            example['judgement'] = int(text[0])
            example['question_1'] = text[1]
            example['question_2'] = text[2]
            example['question_1_words'] = [word for word in example['question_1'].split(' ')]
            example['question_2_words'] = [word for word in example['question_2'].split(' ')]
            example['question_1_tokens'] = [word_to_index_map[word] if word in word_to_index_map.keys() else 0
                                            for word in example['question_1_words']]
            example['question_2_tokens'] = [word_to_index_map[word] if word in word_to_index_map.keys() else 0
                                            for word in example['question_2_words']]
            if n > 0:
                example['question_1_ngrams'] = [word_to_char_ngrams(word, n) if word in word_to_index_map.keys() else [UNKNOWN]
                                                for word in example['question_1_words']]
                example['question_2_ngrams'] = [word_to_char_ngrams(word, n) if word in word_to_index_map.keys() else [UNKNOWN]
                                                for word in example['question_2_words']]
            example['pair_id'] = text[3]
            data.append(example)

            if add_reversed:
                example = {}
                text = line.strip().lower().split('\t')
                example['judgement'] = int(text[0])
                example['question_1'] = text[2]  # reverse q1 and q2
                example['question_2'] = text[1]
                example['question_1_words'] = [word for word in example['question_1'].split(' ')]
                example['question_2_words'] = [word for word in example['question_2'].split(' ')]
                example['question_1_tokens'] = [word_to_index_map[word] if word in word_to_index_map.keys() else 0
                                                for word in example['question_1_words']]
                example['question_2_tokens'] = [word_to_index_map[word] if word in word_to_index_map.keys() else 0
                                                for word in example['question_2_words']]
                if n > 0:
                    example['question_1_ngrams'] = [word_to_char_ngrams(word, n) if word in word_to_index_map.keys() else [UNKNOWN]
                                                    for word in example['question_1_words']]
                    example['question_2_ngrams'] = [word_to_char_ngrams(word, n) if word in word_to_index_map.keys() else [UNKNOWN]
                                                    for word in example['question_2_words']]
                example['pair_id'] = text[3]
                data.append(example)
    return data


def add_char_ngrams(data, build_ngram_map=True, ngram_to_index_map=None):
    if build_ngram_map:
        ngram_to_index_map = {UNKNOWN: 0, PADDING: 1}
        index_to_ngram_map = {0: UNKNOWN, 1: PADDING}
        cur_index = 2
        # build ngram map
        for example in data:
            for ngrams in example['question_1_ngrams'] + example['question_2_ngrams']:
                for ngram in ngrams:
                    if ngram not in ngram_to_index_map.keys():
                        ngram_to_index_map[ngram] = cur_index
                        index_to_ngram_map[cur_index] = ngram
                        cur_index += 1

    # add index for ngram
    for example in data:
        example['question_1_ngrams'] = [[ngram_to_index_map[ngram] if ngram in ngram_to_index_map.keys() else 0 for ngram in ngrams]
                                        for ngrams in example['question_1_ngrams']]
        example['question_2_ngrams'] = [[ngram_to_index_map[ngram] if ngram in ngram_to_index_map.keys() else 0 for ngram in ngrams]
                                        for ngrams in example['question_2_ngrams']]

    if build_ngram_map:
        return data, ngram_to_index_map, index_to_ngram_map
    else:
        return data


def word_to_char_ngrams(word, n=5, max_len=15):
    tmp = '#' + word + '#'
    if len(tmp) < n:
        return [tmp]
    else:
        return [tmp[i: i+n] for i in range(min(len(tmp)-n+1, max_len-n+1))]


def load_embed(path):
    vocabulary = [UNKNOWN, PADDING]
    word_embeddings = [list(np.random.randn(300)), list(np.random.randn(300))]
    word_to_index_map = {UNKNOWN: 0, PADDING: 1}
    index_to_word_map = {0: UNKNOWN, 1: PADDING}
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            item = line.strip().split(' ')
            word = item[0]
            vector = [float(x) for x in item[1:]]
            vocabulary.append(word)
            word_embeddings.append(vector)
            word_to_index_map[word] = i+2
            index_to_word_map[i+2] = word
    return vocabulary, np.array(word_embeddings), word_to_index_map, index_to_word_map


def load_ngram_vocab(path):
    ngram_to_index_map = {UNKNOWN: 0, PADDING: 1}
    index_to_ngram_map = {0: UNKNOWN, 1: PADDING}
    vocabulary = [UNKNOWN, PADDING]
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            ngram = line.strip()
            vocabulary.append(ngram)
            ngram_to_index_map[ngram] = i+2
            index_to_ngram_map[i+2] = ngram
    return vocabulary, ngram_to_index_map, index_to_ngram_map


def batch_iter(dataset, batch_size, use_ngram=False, shuffle=True):
    start = -1 * batch_size
    dataset_size = len(dataset)
    order = list(range(dataset_size))
    if shuffle:
        random.shuffle(order)

    while True:
        start += batch_size
        judgement = []
        question_1 = []
        question_2 = []
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            if shuffle:
                random.shuffle(order)
        batch_indices = order[start:start + batch_size]
        batch = [dataset[index] for index in batch_indices]
        max_length_ngram = 0
        for k in batch:
            judgement.append(k['judgement'])
            if use_ngram:
                question_1.append(k['question_1_ngrams'])
                question_2.append(k['question_2_ngrams'])
                max_length_ngram = max([max_length_ngram,
                                        max([len(ngrams) for ngrams in k['question_1_ngrams']]),
                                        max([len(ngrams) for ngrams in k['question_2_ngrams']])])
            else:
                question_1.append(k['question_1_tokens'])
                question_2.append(k['question_2_tokens'])
        max_length = max([len(question) for question in question_1] + [len(question) for question in question_2])
        if use_ngram:
            for question in question_1:
                question.extend([[1]]*(max_length-len(question)))
                for ngrams in question:
                    ngrams.extend([1]*(max_length_ngram-len(ngrams)))
            for question in question_2:
                question.extend([[1]]*(max_length-len(question)))
                for ngrams in question:
                    ngrams.extend([1]*(max_length_ngram-len(ngrams)))
        else:
            for question in question_1:
                question.extend([1]*(max_length-len(question)))
            for question in question_2:
                question.extend([1]*(max_length-len(question)))
        yield [judgement, question_1, question_2]


if __name__ == '__main__':
    data_dir = '../data'
    vocabulary, word_embeddings, word_to_index_map, index_to_word_map = load_embed(data_dir + '/wordvec.txt')
    training_set = load_data(data_dir + '/dev.tsv', word_to_index_map)  # use dev set for faster test
    x = batch_iter(training_set, 2)
    for item in x:
        print(item)
        print([index_to_word_map[x] for x in item[1][0]])
        print([index_to_word_map[x] for x in item[2][0]])
        break

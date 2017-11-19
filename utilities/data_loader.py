import numpy as np
import random


UNKNOWN = '<UNK>'  # 0
PADDING = '<PAD>'  # 1


def load_data(path, word_to_index_map, add_reversed=False):
    data = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            example = {}
            text = line.strip().lower().split('\t')
            example['judgement'] = int(text[0])
            example['question_1'] = text[1]
            example['question_2'] = text[2]
            example['question_1_tokens'] = [word_to_index_map[word] if word in word_to_index_map.keys() else 0
                                            for word in example['question_1'].split(' ')]
            example['question_2_tokens'] = [word_to_index_map[word] if word in word_to_index_map.keys() else 0
                                            for word in example['question_2'].split(' ')]
            example['pair_id'] = text[3]
            data.append(example)
            if add_reversed:
                example = {}
                text = line.strip().lower().split('\t')
                example['judgement'] = int(text[0])
                example['question_1'] = text[2]  # reverse q1 and q2
                example['question_2'] = text[1]
                example['question_1_tokens'] = [word_to_index_map[word] if word in word_to_index_map.keys() else 0
                                                for word in example['question_1'].split(' ')]
                example['question_2_tokens'] = [word_to_index_map[word] if word in word_to_index_map.keys() else 0
                                                for word in example['question_2'].split(' ')]
                example['pair_id'] = text[3]
                data.append(example)
    return data


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


def batch_iter(dataset, batch_size, shuffle=True):
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
        for k in batch:
            judgement.append(k['judgement'])
            question_1.append(k['question_1_tokens'])
            question_2.append(k['question_2_tokens'])
        max_length = max([len(question) for question in question_1] + [len(question) for question in question_2])
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

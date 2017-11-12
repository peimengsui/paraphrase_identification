import numpy as np
import random


def load_data(path):
    data = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            example = {}
            text = line.strip().lower().split('\t')
            example['judgement'] = int(text[0])
            example['question_1'] = text[1]
            example['question_2'] = text[2]
            example['pair_id'] = text[3]
            data.append(example)
    return data


def load_embed(path):
    PADDING = "<PAD>"
    UNKNOWN = "<UNK>"
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


def batch_iter(dataset, batch_size, word_to_index_map):
    start = -1 * batch_size
    dataset_size = len(dataset)
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        judgement = []
        question_1 = []
        question_2 = []
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)
        batch_indices = order[start:start + batch_size]
        batch = [dataset[index] for index in batch_indices]
        for k in batch:
            judgement.append(k['judgement'])
            question_1.append([word_to_index_map[word] for word in k['question_1'].split(' ')])
            question_2.append([word_to_index_map[word] for word in k['question_2'].split(' ')])
        yield [judgement, question_1, question_2]


if __name__ == '__main__':
    data_dir = '../data'
    training_set = load_data(data_dir + '/train.tsv')[:100]  # subset for faster test
    vocabulary, word_embeddings, word_to_index_map, index_to_word_map = load_embed(data_dir + '/wordvec.txt')
    x = batch_iter(training_set, 1, word_to_index_map)
    for item in x:
        print(item)
        print([index_to_word_map[x] for x in item[1][0]])
        print([index_to_word_map[x] for x in item[2][0]])
        break

""" Input data for word2vec
"""

from collections import defaultdict, deque
import numpy


class InputData:
    """ Load input data and build training samples for word2vec
    """
    def __init__(self, path, min_count=5):
        self.path = path
        self.load_vocabulary(min_count=min_count)
        self.word_pair_catch = deque()
        self.init_sample_table()
        print("Finish input data initiation")
        print("sentence_length", self.sentence_length)
        print("sentence_count", self.sentence_count)

    def load_vocabulary(self, min_count):
        """ Load data, build vocabulary which appear at leat min_count times
        """
        self.input_stream = open(self.path, encoding="utf-8")
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequencies = defaultdict(int)
        for line in self.input_stream:
            self.sentence_count += 1
            words = line.lower().strip().split()
            self.sentence_length += len(words)
            for _word in words:
                word_frequencies[_word] += 1
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        self.word_frequencies = dict()
        for _word, _count in word_frequencies.items():
            if _count < min_count:
                self.sentence_length -= _count
                continue
            self.word2id[_word] = wid
            self.id2word[wid] = _word
            self.word_frequencies[wid] = _count
            wid += 1

    @property
    def word_count(self):
        """ vocabulary size
        """
        return len(self.word2id)

    def init_sample_table(self):
        """ Init sample table for negative samples
        """
        self.sample_table = []
        table_size = 1e8
        pow_freq = numpy.array(list(self.word_frequencies.values()))**0.75
        pow_freq_sum = sum(pow_freq)
        ratio = pow_freq / pow_freq_sum
        count = numpy.round(ratio * table_size)
        for wid, wcount in enumerate(count):
            self.sample_table += [wid] * int(wcount)
        self.sample_table = numpy.array(self.sample_table)

    def get_batch_pairs(self, batch_size, window_size):
        """ Get batch of word id pairs
        """
        while len(self.word_pair_catch) < batch_size:
            sentence = self.input_stream.readline()
            if sentence is None or not sentence:
                self.input_stream = open(self.path, encoding="utf-8")
                sentence = self.input_stream.readline()
            word_ids = []
            for _word in sentence.lower().strip().split():
                if _word not in self.word2id:
                    continue
                word_ids.append(self.word2id[_word])
            for idx1st, wid1st in enumerate(word_ids):
                idx_start = max(idx1st - window_size, 0)
                neighbors = word_ids[idx_start : idx1st + window_size]
                for idx2nd, wid2nd in enumerate(neighbors, start=idx_start):
                    assert wid1st < self.word_count
                    assert wid2nd < self.word_count
                    if idx1st == idx2nd:
                        continue
                    self.word_pair_catch.append((wid1st, wid2nd))
        batch_pairs = [self.word_pair_catch.popleft() for _ in range(batch_size)]
        return batch_pairs

    def get_neg_pairs(self, pos_pairs, count):
        """ Get negative word id pairs
        """
        neg_pairs = numpy.random.choice(
            self.sample_table, size=(len(pos_pairs), count))
        neg_pairs = neg_pairs.tolist()
        return neg_pairs

    def estimate_pair_count(self, window_size):
        """ Estimate total number of pairs
        """
        return (self.sentence_length * (2 * window_size - 1)
                - (self.sentence_count - 1) * (1 + window_size) * window_size)



def test():
    """ Simple grammar test
    """
    data = InputData("./data.txt")
    print("Estimated pair count:", data.estimate_pair_count(window_size=2))
    for idx in range(10):
        print("batch", idx, data.get_batch_pairs(batch_size=5, window_size=2))

if __name__ == "__main__":
    test()

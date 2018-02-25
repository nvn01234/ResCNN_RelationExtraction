import numpy as np
import nltk
import itertools

from gensim.models import KeyedVectors

from .Relation import Relation
from .Sentence import Sentence
from html.parser import HTMLParser
from nltk import word_tokenize

class SemEvalParser(HTMLParser):
    def __init__(self):
        super(SemEvalParser, self).__init__()
        self.data = []
        self.e1 = None
        self.e2 = None
        self.e1pos = 0
        self.e2pos = 0

    def handle_starttag(self, tag, attrs):
        super(SemEvalParser, self).handle_starttag(tag, attrs)
        setattr(self, tag, True)

    def handle_data(self, data):
        super(SemEvalParser, self).handle_data(data)
        data = data.strip()
        if self.e1 is True:
            data = "e1_" + data.replace(" ", "_")
            self.e1 = data
        elif self.e2 is True:
            data = "e2_" + data.replace(" ", "_")
            self.e2 = data
        self.data.append(data)

    def tokenize(self):
        tokens = word_tokenize(" ".join(self.data))
        for i, w in enumerate(tokens):
            if w == self.e1:
                self.e1pos = i
            if w == self.e2:
                self.e2pos = i
        self.tokens = [t[3:] if t.startswith("e1_") or t.startswith("e2_") else t for t in tokens]
        self.e1 = self.e1[3:]
        self.e2 = self.e2[3:]

class DataManager:
    def __init__(self, sequence_length):
        self.wordvector_dim = 0
        self.sequence_length = sequence_length
        self.word2index = {}
        self.index2vector = []
        self.relations = {}
        self.training_data = []
        self.testing_data = []
        self.load_word2vec()
        self.load_relations()

    def load_word2vec(self):
        #load word2vec from file
        #Two data structure: word2index, index2vector
        wordvector = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)
        self.wordvector_dim = 300
        self.word2index["PAD"] = 0
        self.word2index["UNK"] = 1
        self.index2vector.append(np.zeros(self.wordvector_dim))
        self.index2vector.append(np.random.uniform(-0.25, 0.25, self.wordvector_dim))
        for w in wordvector.index2word:
            self.word2index[w] = len(self.word2index)
            self.index2vector.append(wordvector.word_vec(w))

        print("WordTotal=\t", len(self.index2vector))
        print("Word dimension=\t", self.wordvector_dim)

    def load_relations(self):
        #load relation from file
        relation_data = {'Other': 0,
                         'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
                         'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
                         'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
                         'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
                         'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
                         'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
                         'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
                         'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
                         'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}
        for k, v in relation_data.items():
            r = Relation(k, v)
            self.relations[k] = r
        for r in self.relations:
            self.relations[r].generate_vector(len(self.relations))
        print("RelationTotal: "+str(len(self.relations)))

    def load_training_data(self, filename="data/TRAIN_FILE.TXT"):
        #load training data from file
        print("Start loading training data.")
        print("====================")
        with open(filename, "r", encoding="utf8") as f:
            training_data = f.read().strip().split("\n\n")
        for data in training_data:
            line1, r, _ = data.strip().split("\n")
            if r in self.relations:
                r = self.relations[r]
            else:
                r = self.relations["NA"]
            text = line1.strip().split("\t")[1][1:-1]
            parser = SemEvalParser()
            parser.feed(text)
            parser.tokenize()
            s = Sentence(
                parser.e1,
                parser.e2,
                r,
                parser.tokens
            )
            self.training_data.append(s)

        return self.training_data

    def load_testing_data(self):
        #load training data from file
        print("Start loading testing data.")
        print("====================")
        with open("data/TEST_FILE_FULL.TXT", "r", encoding="utf8") as f:
            testing_data = f.read().strip().split("\n\n")
        #for data in testing_data:
        for data in testing_data:
            line1, r, _ = data.strip().split("\n")
            if r in self.relations:
                r = self.relations[r]
            else:
                r = self.relations["NA"]
            text = line1.strip().split("\t")[1][1:-1]
            parser = SemEvalParser()
            parser.feed(text)
            parser.tokenize()
            s = Sentence(
                parser.e1,
                parser.e2,
                r,
                parser.tokens
            )
            self.testing_data.append(s)
        return self.testing_data

    def relation_analyze(self):
        for r in self.relations:
            print(r+": "+str(self.relations[r].number))

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            #Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                if start_index == end_index:
                    continue
                else:
                    yield shuffled_data[start_index:end_index]

    def generate_x(self, data):
        x = []
        for d in data:
            v = []
            words = d.words
            e1 = d.entity1
            e2 = d.entity2
            for i, w in enumerate(words):
                w = w.split("_")
                tmp = []
                for _w in w:
                    if _w not in self.word2index:
                        tmp.append(self.index2vector[self.word2index["UNK"]])
                    else:
                        tmp.append(self.index2vector[self.word2index[_w]])
                tmp = np.average(w, axis=0)
                v.append(tmp)
            vectors = self.padding(v)
            x.append(vectors)
        return x

    def generate_y(self, data):
        return [d.relation.vector for d in data]

    def generate_p(self, data):
        p1 = []
        p2 = []
        for d in data:
            p11 = []
            p22 = []
            e1 = d.entity1
            e2 = d.entity2
            words = d.words
            l1 = 0
            l2 = 0
            for i, w in enumerate(words):
                if w == e1:
                    l1 = i
                if w == e2:
                    l2 = i
            for i, w in enumerate(words):
                a = i-l1
                b = i-l2
                if a > 30:
                    a = 30
                if b > 30:
                    b = 30
                if a < -30:
                    a = -30
                if b < -30:
                    b = -30
                p11.append(a+31)
                p22.append(b+31)
            a = self.sequence_length-len(p11)
            if a > 0:
                front = int(a/2)
                back = a-front
                front_vec = [0 for i in range(front)]
                back_vec = [0 for i in range(back)]
                p11 = front_vec + p11 + back_vec
                p22 = front_vec + p22 + back_vec
            else:
                p11 = p11[:self.sequence_length]
                p22 = p22[:self.sequence_length]
            p1.append(p11)
            p2.append(p22)
        return p1, p2


    def padding(self, vectors):
        a = self.sequence_length-len(vectors)
        if a > 0:
            front = int(a/2)
            back = a-front
            front_vec = [np.zeros(self.wordvector_dim) for i in range(front)]
            back_vec = [np.zeros(self.wordvector_dim) for i in range(back)]
            vectors = front_vec + vectors + back_vec
        else:
            vectors = vectors[:self.sequence_length]
        return vectors

    def word2num(self, words):
        return [self.word2index[w] for w in words]

def __init__():
    return 0

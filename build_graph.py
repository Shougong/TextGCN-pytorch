import os
import random
from nltk.util import pr
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import clean_str # loadWord2Vec
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

from config import Config

class BuildGraph(object):
    def __init__(self, config, dataset) -> None:
        super(BuildGraph, self).__init__()
        self.config = config    # data file path
        self.dataset = dataset

        self.is_cosine = False
        self.window_size = 20
        self.word_embedding_dim = 300
        self.word_vector_map = {}

        print("=" * 20, " load data ", "=" * 20)
        self.load_data()
        print("num of document: {0}".format(len(self.shuffle_doc_name_list)))
        print("train size: {0}, dev size: {1}, test size: {2}".format(self.train_size, self.dev_size, self.test_size))
        print("work done.")        

        print("=" * 20, " build vocab ", "=" * 20)
        self.vocab, self.vocab_size, self.word_freq, self.word2id = self.build_vocab()
        self.node_size = self.train_size + self.dev_size + self.vocab_size + self.test_size
        print("vocab size: {0}, node size: {1}".format(self.vocab_size, self.node_size))
        print("work done.")

        print("=" * 20, " build word doc list ", "=" * 20)
        self.word_doc_freq, self.word_doc_list = self.build_wod_doc_list()
        print("size of word_doc_freq: {0}".format(len(self.word_doc_freq)))
        print("size of word_doc_list: {0}".format(len(self.word_doc_list)))
        print("work done.")

        print("=" * 20, " build label list ", "=" * 20)
        self.label_list = self.build_label_list()
        print("num of label: {0}".format(len(self.label_list)))
        print(self.label_list)
        print("work done.")

        print("=" * 20, " build nodes ", "=" * 20)
        self.x, self.y, self.tx, self.ty, self.allx, self.ally = self.build_nodes()
        print(f"shape of x: {self.x.shape}, shape of y: {self.y.shape}")
        print(f"shape of tx: {self.tx.shape}, shape of ty: {self.ty.shape}")
        print(f"shape of allx: {self.allx.shape}, shape of ally: {self.ally.shape}")
        print("work done.")

        print("=" * 20, " build edges ", "=" * 20)
        self.adj = self.build_edges()

        print("=" * 20, " save parameters ", "=" * 20)
        self.save_data()
        print("work done.")
    
    def load_data(self):
        # 加载shuffle后的数据
        shuffle_doc_words_f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + "_doc_words_shuffle.txt", 'r')
        self.shuffle_doc_words_list = shuffle_doc_words_f.readlines()
        shuffle_doc_words_f.close()
        # 加载shuffle后的文档名
        shuffle_doc_name_f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + "_doc_name_shuffle.txt", 'r')
        self.shuffle_doc_name_list = shuffle_doc_name_f.readlines()
        shuffle_doc_name_f.close()
        # 加载测试集
        test_index_f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + ".test.index", 'r')
        self.test_size = len(test_index_f.readlines())
        test_index_f.close()        
        # 加载训练集
        train_index_f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + ".train.index", 'r')
        train_ids = train_index_f.readlines()
        train_index_f.close()
        self.dev_size = int(self.config.train_dev * len(train_ids))
        self.train_size = len(train_ids) - self.dev_size
        
    def build_vocab(self):
        word_freq = {}
        word_set = set()
        for doc_words in self.shuffle_doc_words_list:
            words = doc_words.strip().split()
            for word in words:
                word_set.add(word)
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        vocab = list(word_set)
        vocab_size = len(vocab)
        
        word2id = {}
        for i in range(vocab_size):
            word2id[vocab[i]] = i
        return vocab, vocab_size, word_freq, word2id
    
    def build_wod_doc_list(self):
        word_doc_list = {}

        for i in range(len(self.shuffle_doc_words_list)):
            doc_words = self.shuffle_doc_words_list[i]
            words = doc_words.split()
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                if word in word_doc_list:
                    doc_list = word_doc_list[word]
                    doc_list.append(i)
                    word_doc_list[word] = doc_list
                else:
                    word_doc_list[word] = [i]
                appeared.add(word)

        word_doc_freq = {}
        for word, doc_list in word_doc_list.items():
            word_doc_freq[word] = len(doc_list)
        return word_doc_freq, word_doc_list
    
    def build_label_list(self):
        label_set = set()
        for doc_meta in self.shuffle_doc_name_list:
            temp = doc_meta.strip().split('\t')
            label_set.add(temp[2])
        return list(label_set)
    
    def build_nodes(self):
        row_x = []
        col_x = []
        data_x = []
        for i in range(self.train_size):
            doc_vec = np.array([0.0 for k in range(self.word_embedding_dim)])
            doc_words = self.shuffle_doc_words_list[i]
            words = doc_words.strip().split()
            doc_len = len(words)
            for word in words:
                if word in self.word_vector_map:
                    word_vector = self.word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)
            
            for j in range(self.word_embedding_dim):
                row_x.append(i)
                col_x.append(j)
                data_x.append(doc_vec[j] / doc_len)

        x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
            self.train_size, self.word_embedding_dim))

        y = []
        for i in range(self.train_size):
            doc_meta = self.shuffle_doc_name_list[i]
            temp = doc_meta.strip().split('\t')
            label = temp[2]
            one_hot = [0 for l in range(len(self.label_list))]
            label_index = self.label_list.index(label)
            one_hot[label_index] = 1
            y.append(one_hot)
        y = np.array(y)

        # tx: feature vectors of test docs, no initial features
        row_tx = []
        col_tx = []
        data_tx = []
        for i in range(self.test_size):
            doc_vec = np.array([0.0 for k in range(self.word_embedding_dim)])
            doc_words = self.shuffle_doc_words_list[i + self.train_size + self.dev_size]
            words = doc_words.strip().split()
            doc_len = len(words)
            for word in words:
                if word in self.word_vector_map:
                    word_vector = self.word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(self.word_embedding_dim):
                row_tx.append(i)
                col_tx.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

        # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
        tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                           shape=(self.test_size, self.word_embedding_dim))

        ty = []
        for i in range(self.test_size):
            doc_meta = self.shuffle_doc_name_list[i + self.train_size + self.dev_size]
            temp = doc_meta.strip().split('\t')
            label = temp[2]
            one_hot = [0 for l in range(len(self.label_list))]
            label_index = self.label_list.index(label)
            one_hot[label_index] = 1
            ty.append(one_hot)
        ty = np.array(ty)

        # allx: the the feature vectors of both labeled and unlabeled training instances
        # (a superset of x)
        # unlabeled training instances -> words

        word_vectors = np.random.uniform(-0.01, 0.01,
                                         (self.vocab_size, self.word_embedding_dim))

        for i in range(len(self.vocab)):
            word = self.vocab[i]
            if word in self.word_vector_map:
                vector = self.word_vector_map[word]
                word_vectors[i] = vector

        row_allx = []
        col_allx = []
        data_allx = []

        for i in range(self.train_size + self.dev_size):
            doc_vec = np.array([0.0 for k in range(self.word_embedding_dim)])
            doc_words = self.shuffle_doc_words_list[i]
            words = doc_words.strip().split()
            doc_len = len(words)
            for word in words:
                if word in self.word_vector_map:
                    word_vector = self.word_vector_map[word]
                    doc_vec = doc_vec + np.array(word_vector)

            for j in range(self.word_embedding_dim):
                row_allx.append(int(i))
                col_allx.append(j)
                # np.random.uniform(-0.25, 0.25)
                data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
        for i in range(self.vocab_size):
            for j in range(self.word_embedding_dim):
                row_allx.append(int(i + self.train_size + self.dev_size))
                col_allx.append(j)
                data_allx.append(word_vectors.item((i, j)))

        row_allx = np.array(row_allx)
        col_allx = np.array(col_allx)
        data_allx = np.array(data_allx)

        allx = sp.csr_matrix(
            (data_allx, (row_allx, col_allx)), shape=(self.train_size +  self.dev_size + self.vocab_size, self.word_embedding_dim))

        ally = []
        for i in range(self.train_size + self.dev_size):
            doc_meta = self.shuffle_doc_name_list[i]
            temp = doc_meta.strip().split('\t')
            label = temp[2]
            one_hot = [0 for l in range(len(self.label_list))]
            label_index = self.label_list.index(label)
            one_hot[label_index] = 1
            ally.append(one_hot)

        for i in range(self.vocab_size):
            one_hot = [0 for l in range(len(self.label_list))]
            ally.append(one_hot)

        ally = np.array(ally)
        return x, y, tx, ty, allx, ally

    def build_edges(self):
        '''
        build edges by pmi and tf-idf
        '''
        self.row = []
        self.col = []
        self.weight = []
        windows = self.get_windows()
        word_window_freq = self.get_word_window_freq(windows)
        word_pair_count = self.get_word_pair_count(windows)
        if self.is_cosine:
            self.compute_cosine_similarity_as_edge_weight()
        else:
            self.compute_pmi_as_edge_weight(windows, word_pair_count, word_window_freq)
        self.compute_doc_word_freq_as_edge_weight()
        
        adj = sp.csr_matrix((self.weight, (self.row, self.col)), shape=(self.node_size, self.node_size))
        print(f"shape of adj: {adj.shape}")
        print("build edges work done.")
        return adj

    def get_windows(self):
        '''
        获得划分的window
        '''
        # word co-occurence with context windows
        windows = []

        for doc_words in self.shuffle_doc_words_list:
            words = doc_words.strip().split()
            length = len(words)
            if length <= self.window_size:
                windows.append(words)
            else:
                # print(length, length - window_size + 1), step = 1
                for j in range(length - self.window_size + 1):
                    window = words[j: j + self.window_size]
                    windows.append(window)
        print(f"get windows have done, num of windows: {len(windows)}")
        return windows
    
    def get_word_window_freq(self, windows):
        '''
        计算#W(i)
        '''
        word_window_freq = {}
        for window in windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:   # 对出现过的词, pass
                    continue
                if window[i] in word_window_freq:
                    word_window_freq[window[i]] += 1
                else:
                    word_window_freq[window[i]] = 1
                appeared.add(window[i])
        print(f"get word_window_freq have done, size of word_window_freq: {len(word_window_freq)}")
        return word_window_freq

    def get_word_pair_count(self, windows):
        word_pair_count = {}        # 计算# w(i, j), 其中i, j是单词
        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = self.word2id[word_i]
                    word_j = window[j]
                    word_j_id = self.word2id[word_j]
                    if word_i_id == word_j_id:  # 相同word采取pass, 这一步的#W(i, i) = 1操作是在utils的方法中构建的
                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)   # i_id, j_id形式构造word_pair
                    if word_pair_str in word_pair_count:    # 统计word_pair的出现次数
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders                            # 因为是无向的, 因此反过来也需要统计一遍
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
        print(f"size of word_pair_count: {len(word_pair_count)}")
        print("compute word pair count have done.")
        return word_pair_count

    def compute_pmi_as_edge_weight(self, windows, word_pair_count, word_window_freq):
        # pmi as weights
        # #W, total number of sliding windows in the corpus
        num_window = len(windows)           

        for key in word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            # #W(i, j), the number of sliding windows that contains both word i and j
            count = word_pair_count[key]    
            # #W(i) & #W(j), the number of sliding windows in a corpus that contains word i
            word_freq_i = word_window_freq[self.vocab[i]]   
            word_freq_j = word_window_freq[self.vocab[j]]   
            pmi = log((1.0 * count / num_window) /
                      (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
            if pmi <= 0:
                continue
            self.row.append(self.train_size + self.dev_size + i)
            self.col.append(self.train_size + self.dev_size + j)
            self.weight.append(pmi)

    def compute_cosine_similarity_as_edge_weight(self):
        # word vector cosine similarity as weights
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                if self.vocab[i] in self.word_vector_map and self.vocab[j] in self.word_vector_map:
                    vector_i = np.array(self.word_vector_map[self.vocab[i]])
                    vector_j = np.array(self.word_vector_map[self.vocab[j]])
                    similarity = 1.0 - cosine(vector_i, vector_j)
                    if similarity > 0.9:
                        print(self.vocab[i], self.vocab[j], similarity)
                        self.row.append(self.train_size + self.dev_size + i)
                        self.col.append(self.train_size + self.dev_size + j)
                        self.weight.append(similarity)

    def compute_doc_word_freq_as_edge_weight(self):
        # doc word frequency
        doc_word_freq = {}

        for doc_id in range(len(self.shuffle_doc_words_list)):
            doc_words = self.shuffle_doc_words_list[doc_id]
            words = doc_words.split()
            for word in words:
                word_id = self.word2id[word]
                doc_word_str = str(doc_id) + ',' + str(word_id)
                if doc_word_str in doc_word_freq:
                    doc_word_freq[doc_word_str] += 1
                else:
                    doc_word_freq[doc_word_str] = 1

        for i in range(len(self.shuffle_doc_words_list)):
            doc_words = self.shuffle_doc_words_list[i]
            words = doc_words.split()
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                j = self.word2id[word]
                key = str(i) + ',' + str(j)
                freq = doc_word_freq[key]
                if i < self.train_size + self.dev_size:
                    self.row.append(i)
                else:
                    self.row.append(i + self.vocab_size)
                self.col.append(self.train_size + self.dev_size + j)
                idf = log(1.0 * len(self.shuffle_doc_words_list) /
                          self.word_doc_freq[self.vocab[j]])
                self.weight.append(freq * idf)
                doc_word_set.add(word)

    def save_data(self):
        f = open(self.config.corpus + "/" + self.dataset + "/" + "ind.{}.label_list".format(self.dataset), 'wb')
        pkl.dump(self.label_list, f)
        f.close
        print("saved adj data.")

        f = open(self.config.corpus + "/" + self.dataset + "/" + "ind.{}.x".format(self.dataset), 'wb')
        pkl.dump(self.x, f)
        f.close
        
        f = open(self.config.corpus + "/" + self.dataset + "/" + "ind.{}.y".format(self.dataset), 'wb')
        pkl.dump(self.y, f)
        f.close
        print("saved train data.")

        f = open(self.config.corpus + "/" + self.dataset + "/" + "ind.{}.tx".format(self.dataset), 'wb')
        pkl.dump(self.tx, f)
        f.close
        f = open(self.config.corpus + "/" + self.dataset + "/" + "ind.{}.ty".format(self.dataset), 'wb')
        pkl.dump(self.ty, f)
        f.close
        print("saved test data.")

        f = open(self.config.corpus + "/" + self.dataset + "/" + "ind.{}.allx".format(self.dataset), 'wb')
        pkl.dump(self.allx, f)
        f.close
        f = open(self.config.corpus + "/" + self.dataset + "/" + "ind.{}.ally".format(self.dataset), 'wb')
        pkl.dump(self.ally, f)
        f.close
        print("saved train & dev data.")

        f = open(self.config.corpus + "/" + self.dataset + "/" + "ind.{}.adj".format(self.dataset), 'wb')
        pkl.dump(self.adj, f)
        f.close
        print("saved adj data.")


if __name__ == "__main__":
    config = Config()
    dataset = '20ng'
    g = BuildGraph(config, dataset)
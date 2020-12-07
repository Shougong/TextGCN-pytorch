from nltk import data
from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

import os
import random

from config import Config
from utils import clean_str

class data_process(object):
    def __init__(self, config, dataset):
        super(data_process, self).__init__()
        self.config = config
        self.dataset = dataset

        print("=" * 20, " load stop_words ", "=" * 20)
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        print("work done.")

        print("=" * 20, " build doc index file ", "=" * 20)
        self.build_doc_index()

        print("=" * 20, " build data ", "=" * 20)
        self.build_data()
        print("work done.")

        print("=" * 20, " prepare data ", "=" * 20)
        self.prepare_data()
        print("work done.")

        print("=" * 20, " shuffle data ", "=" * 20)
        self.train_ids, self.test_ids = self.shuffle_data()
        print("work done.")
    
    def build_doc_index(self):
        assert os.path.exists(self.config.data + "/" + self.dataset), f"{self.dataset} dataset doesn't exist."
        
        f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + "_document.txt", "w+", encoding="utf-8")
        # process data
        dataset_path = self.config.data + "/" + self.dataset
        data_list = os.listdir(dataset_path)
        for data_path in data_list:
            labels = os.listdir(dataset_path + "/" + data_path)
            for label in labels:
                file_list = os.listdir(dataset_path + "/" + data_path + "/" + label)
                for file in file_list:
                    file_path = dataset_path + "/" + data_path + "/" + label + "/" + file
                    string = file_path + "\t" + data_path +"\t" + label + "\n"
                    f.write(string)
        f.close()
        print("work done.")
    
    def build_data(self):
        f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + "_document.txt", 'r')
        lines = f.readlines()
        docs = []
        num_sample = 0
        for doc in lines:
            doc_info = doc.strip().split("\t")
            doc_f = open(doc_info[0], 'rb')
            doc_content = doc_f.read().decode('latin1')     # 编码格式为latin1
            doc_f.close()
            doc_content = doc_content.replace('\n', ' ')
            docs.append(doc_content)
        num_sample = len(docs)
        corpus = '\n'.join(docs)
        f.close()

        f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + ".txt", 'w', encoding='latin1')
        f.write(corpus)
        f.close()
        print("num of samples is {0}".format(num_sample))
    
    def prepare_data(self):
        doc_content_list = []
        f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + ".txt", 'rb')
        lines = f.readlines()
        for line in lines:
            doc_content_list.append(line.strip().decode('latin1'))
        f.close()
        # build word_freq dict
        word_freq = {}
        for doc_content in doc_content_list:
            temp = clean_str(doc_content)
            words = temp.split()
            for word in words:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        
        # build clean_docs
        clean_docs = []
        for doc_content in doc_content_list:
            temp = clean_str(doc_content)
            words = temp.split()
            doc_words = []
            for word in words:
                if word not in self.stop_words and word_freq[word] >= 5:
                    doc_words.append(word)
            doc_str = ' '.join(doc_words).strip()
            clean_docs.append(doc_str)
        clean_corpus_str = '\n'.join(clean_docs)
        f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + ".clean.txt", 'w')
        f.write(clean_corpus_str)
        f.close()

        # statistic seq len
        min_len = 10000
        avg_len = 0
        max_len = 0
        f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + ".clean.txt", 'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            avg_len += len(line)
            min_len = min(min_len, len(line))
            max_len = max(max_len, len(line))
        avg_len /= len(lines)
        print("min len: {0}, max_len: {1}, avg_len: {2}".format(min_len, max_len, avg_len))
    
    def shuffle_data(self):
        doc_name_list = []
        doc_train_list = []
        doc_test_list = []

        f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + "_document.txt", 'r')
        lines = f.readlines()
        for line in lines:
            doc_name_list.append(line.strip())
            temp = line.split("\t")
            if temp[1].find('test') != -1:
                doc_test_list.append(line.strip())
            elif temp[1].find('train') != -1:
                doc_train_list.append(line.strip())
        f.close()
        print("num of train doc: {0}, num of test doc: {1}".format(len(doc_train_list), len(doc_test_list)))
        doc_content_list = []
        f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + ".clean.txt", "r")
        lines = f.readlines()
        for line in lines:
            doc_content_list.append(line.strip())
        f.close()
        # 对训练集构建index
        train_ids = []
        for train_name in doc_train_list:
            train_id = doc_name_list.index(train_name)
            train_ids.append(train_id)
        random.shuffle(train_ids)       # 打乱训练集数据
        # 存储索引化后的数据
        train_ids_str = '\n'.join(str(index) for index in train_ids)
        f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + ".train.index", 'w')
        f.write(train_ids_str)
        f.close()

        # 对测试集构建index
        test_ids = []
        for test_name in doc_test_list:
            test_id = doc_name_list.index(test_name)
            test_ids.append(test_id)
        random.shuffle(test_ids)       # 打乱测试集数据
        test_ids_str = '\n'.join(str(index) for index in test_ids)
        f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + ".test.index", 'w')
        f.write(test_ids_str)
        f.close()

        ids = train_ids + test_ids
        
        shuffle_doc_name_list = []
        shuffle_doc_words_list = []
        for id in ids:
            shuffle_doc_name_list.append(doc_name_list[int(id)])
            shuffle_doc_words_list.append(doc_content_list[int(id)])
        shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
        shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

        f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + "_doc_name_shuffle.txt", 'w')
        f.write(shuffle_doc_name_str)
        f.close()

        f = open(self.config.corpus + "/" + self.dataset + "/" + self.dataset + "_doc_words_shuffle.txt", 'w')
        f.write(shuffle_doc_words_str)
        f.close()
        
        return train_ids, test_ids








if __name__ == "__main__":
    config = Config()
    dataset = '20ng'
    dp = data_process(config, dataset)
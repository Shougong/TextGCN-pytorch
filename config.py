class Config(object):
    def __init__(self) -> None:
        super(Config, self).__init__()
        # 数据目录
        self.data = "data"
        self.corpus = "data/corpus"
        
        # 词汇表
        self.vocab_file = "saved/20ng/vocab.pkl"

        # 数据集划分
        self.train_dev = 0.1

        self.saved = "saved"

        

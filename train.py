import time
from scipy.sparse import data
import torch
import torch.nn.functional as F
from sklearn import metrics
from utils import *
# from models import GCN, MLP
import random
import os
import sys

from models.TextGCN import GCN

from config import Config

def train(model, lr, y_train, train_mask, y_val, val_mask):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(features, adj)
        outputs = outputs[train_mask, :]
        train_labels = y_train[train_mask, :]
        _, target = train_labels.max(dim=1)
        loss = F.cross_entropy(outputs, target.to(device))
        loss.backward()
        optimizer.step()
        train_acc = accuracy(outputs, target.to(device))
        
        if epoch % 10 == 0:
            print("{0} / {1} epochs, train loss: {2}, train acc: {3}".format(epoch, epochs, loss, train_acc), end=', ')
            eval(model, y_val, val_mask)

def eval(model, y_val, val_mask, is_test=False):
    model.eval()
    with torch.no_grad():
        outputs = model(features, adj)
        dev_outputs = outputs[val_mask, :]
        dev_labels = y_val[val_mask, :]
        _, dev_target = dev_labels.max(dim=1)
        dev_loss = F.cross_entropy(dev_outputs, dev_target)
        dev_acc = accuracy(dev_outputs, dev_target)
        if is_test:
            out_str = "test loss: {0}, test acc: {1}"
        else:
            out_str = "dev loss: {0}, dev acc: {1}"
        print(out_str.format(dev_loss, dev_acc))


if __name__ == "__main__":
    dataset = '20ng'
    config = Config()

    # Set random seed
    # np.random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
    
    features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    features = preprocess_features(features)    # feature of nodes
    adj = preprocess_adj(adj)                   # adjacency matrix

    nfeat_dim = features.shape[0]
    
    labels = pkl.load(open(config.corpus + "/" + dataset + "/" + "ind.{}.label_list".format(dataset), 'rb'))
    nclass = len(labels)
    print(f"num of class: {nclass}")

    epochs = 200
    lr = 0.02
    
    nhid = 200
    dropout = 0.5
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 数据转换
    adj = adj.to(device)
    features = features.to(device)
    train_mask = torch.from_numpy(train_mask).to(device)
    val_mask = torch.from_numpy(val_mask).to(device)
    test_mask = torch.from_numpy(test_mask).to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    model = GCN(nfeat=nfeat_dim, nhid=nhid, nclass=nclass, dropout=dropout).to(device)
    start_time = time.time()
    train(model, lr, y_train, train_mask, y_val, val_mask)
    end_time = time.time()
    train_time = end_time - start_time
    print(f"Train stages have done, cost time: {train_time}")
    start_time = time.time()
    eval(model, y_test, test_mask, is_test=True)
    end_time = time.time()
    test_time = end_time - start_time     
    print(f"Test stages have done, cost time: {test_time}")  
    pkl.dump(model, open(config.saved + "/" + dataset + "/model.pkl", 'wb')) 





    


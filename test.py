import nltk
from nltk.corpus import stopwords
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import clean_str
import numpy as np
import scipy.sparse as sp
import sys

# row  = np.array([0, 0, 1, 3, 1, 0, 0])
# col  = np.array([0, 2, 1, 3, 1, 0, 0])
# data = np.array([1, 1, 1, 1, 1, 1, 1])
# coo_mat= sp.coo_matrix((data, (row, col)), shape=(4, 4))
# print(coo_mat.toarray())
# lil_mat = coo_mat.tolil()
# print(lil_mat.toarray())
# print(lil_mat.data)

# def sample_mask(idx, l):
#     """Create mask."""
#     mask = np.zeros(l)
#     mask[idx] = 1
#     return np.array(mask, dtype=np.bool)

# a = np.arange(0, 100)
# labels = a.reshape(10, 10)
# print(labels)

# test_index = range(0, 5)
# test_mask = sample_mask(test_index, labels.shape[0])
# print(test_mask)

# y_train = np.zeros(labels.shape)
# y_train[test_mask, :] = labels[test_mask, :]
# print(y_train)

features = sp.identity(5)
print(features.sum(1))

rowsum = np.array(features.sum(1))

r_inv = np.power(rowsum, -1).flatten()
r_inv[np.isinf(r_inv)] = 0.
r_mat_inv = sp.diags(r_inv)
features = r_mat_inv.dot(features)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            print("is coo ?")
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

features = sparse_to_tuple(features)
import os.path
import numpy as np
from embeddings import Embeddings
import io
import pdb
from data_loading.data_utils import pickle_call,pickle_dump
"""
THIS IS A CHILD OF EMBEDDINGS!
"""

FASTTEXT_NAMES = ['FastText-Crawl', 'FastText-Wiki']

class FastTextEmbeddings(Embeddings):

    def __init__(self, name):
        """
        This class calls the FastText data
        """

        # Load the super class
        super(FastTextEmbeddings, self).__init__()

        # check if the FastText Data exisits
        self.name = name

        if self.name == 'FastText-Crawl':
            self.path = '../data/embeddings/crawl-300d-2M.vec'
            if not os.path.isfile(self.path):
                self.path = 'data/embeddings/crawl-300d-2M.vec'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "please load FastText Embeddings from https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip and store in data/embeddings/")

        elif self.name == 'FastText-Wiki':
            self.path = '../data/embeddings/wiki-news-300d-1M-subword.vec'
            if not os.path.isfile(self.path):
                self.path = 'data/embeddings/wiki-news-300d-1M-subword.vec'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "please load FastText Embeddings from https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip and store in data/embeddings/")

    def parse_line(self, line):
        """
        Here we parse the line of the  embedding file and return the term and the embeddings
        :param line:
        :return:
        """
        elems = line.split()
        term = elems[0]
        embedding = [float(i) for i in elems[1:]]
        return term, embedding

    def load_top_k(self, K, preload=False):
        """
        Option for loading strategy: Only load top k of embeddings assuming that they are ordered in frequency
        :param K: Number of top k embeddings to be retrieved
        :return:embeddings matrix as numpy
        """

        # This embedding dataset does not have PAD UNK START and END tokens pretrained that is why we initialize them
        # ourselves and only load K - 4 embeddings

        K = K - 4

        embeddings = []
        with open(self.path, 'r') as f:
            for k, line in enumerate(f):
                if k == 0: continue
                if k > K: break
                term, embedding = self.parse_line(line)
                self.add_term(term, preload=preload)
                embeddings.append(embedding)
        special_embeddings = self.add_special_embeddings(len(embeddings[0]), preload=preload)
        if special_embeddings != []:
            embeddings = embeddings + special_embeddings

        return np.array(embeddings)

    def get_name(self):
        return self.name




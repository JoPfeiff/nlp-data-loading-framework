import os.path
import numpy as np
from embeddings import Embeddings
from data_loading.data_utils import pickle_call

POLYGLOT_NAMES = ['Polyglot']

"""
THIS IS A CHILD OF EMBEDDINGS!

THIS SCRIPT SHOULD BE USED TO INITIALIZE A NEW EMBEDDING DATASET
"""

class PolyglotEmbeddings(Embeddings):

    def __init__(self):
        """
        This class calls the FastText data
        """

        # Load the super class
        super(PolyglotEmbeddings, self).__init__()

        # Name of the embedding dataset
        self.name = 'Polyglot'

        path = '../data/embeddings/polyglot-en.pkl'
        if not os.path.isfile(path):
            path = 'data/embeddings/polyglot-en.pkl'
            if not os.path.isfile(path):
                raise Exception(
                    "please load Polyglot Embeddings from http://bit.ly/19bSoAS and store in data/embeddings/")

        self.path = path
        self.poly_data = pickle_call(self.path)


    def load_top_k(self, K, preload=False):
        """
        Option for loading strategy: Only load top k of embeddings assuming that they are ordered in frequency
        :param K: Number of top k embeddings to be retrieved
        :return:embeddings matrix as numpy
        """

        vocab_list = self.poly_data[0]
        embeddings = self.poly_data[1]
        K = min(K,len(embeddings))
        for i in range(K):
            self.add_term(vocab_list[i], preload=preload)

        special_embeddings = self.add_special_embeddings(len(embeddings[0]), preload=preload)
        if special_embeddings != []:
            embeddings = embeddings + special_embeddings

        return embeddings[:K]


    def get_name(self):
        return self.name

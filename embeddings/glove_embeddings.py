import os.path
import numpy as np
from embeddings import Embeddings
import io
import pdb
from data_loading.data_utils import pickle_call,pickle_dump
"""
THIS IS A CHILD OF EMBEDDINGS!
"""

GLOVE_NAMES = [ 'Glove-Twitter-25', 'Glove-Twitter-50', 'Glove-Twitter-100', 'Glove-Twitter-200', 'Glove-Common-42B-300'
                  'Glove-Common-840B-300', 'Glove-Wiki-50', 'Glove-Wiki-100', 'Glove-Wiki-200', 'Glove-Wiki-300']

class GloveEmbeddings(Embeddings):

    def __init__(self, name):
        """
        This class calls the FastText data
        """

        # Load the super class
        super(GloveEmbeddings, self).__init__()

        # check if the FastText Data exisits
        self.name = name

        if self.name == 'Glove-Twitter-25':
            self.path = '../data/embeddings/glove.twitter.27B.25d.txt'
            if not os.path.isfile(self.path):
                self.path = 'data/embeddings/glove.twitter.27B.25d.txt'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "please load Glove Embeddings from http://nlp.stanford.edu/data/glove.twitter.27B.zip and store in data/embeddings/")

        elif self.name == 'Glove-Twitter-50':
            self.path = '../data/embeddings/glove.twitter.27B.50d.txt'
            if not os.path.isfile(self.path):
                self.path = 'data/embeddings/glove.twitter.27B.50d.txt'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "please load Glove Embeddings from http://nlp.stanford.edu/data/glove.twitter.27B.zip and store in data/embeddings/")

        elif self.name == 'Glove-Twitter-100':
            self.path = '../data/embeddings/glove.twitter.27B.100d.txt'
            if not os.path.isfile(self.path):
                self.path = 'data/embeddings/glove.twitter.27B.100d.txt'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "please load Glove Embeddings from http://nlp.stanford.edu/data/glove.twitter.27B.zip and store in data/embeddings/")

        elif self.name == 'Glove-Twitter-200':
            self.path = '../data/embeddings/glove.twitter.27B.200d.txt'
            if not os.path.isfile(self.path):
                self.path = 'data/embeddings/glove.twitter.27B.200d.txt'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "please load Glove Embeddings from http://nlp.stanford.edu/data/glove.twitter.27B.zip and store in data/embeddings/")

        elif self.name == 'Glove-Common-42B-300':
            self.path = '../data/embeddings/glove.42B.300d.txt'
            if not os.path.isfile(self.path):
                self.path = 'data/embeddings/glove.42B.300d.txt'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "please load Glove Embeddings from http://nlp.stanford.edu/data/glove.42B.300d.zip and store in data/embeddings/")

        elif self.name == 'Glove-Common-840B-300':
            self.path = '../data/embeddings/glove.840B.300d.txt'
            if not os.path.isfile(self.path):
                self.path = 'data/embeddings/glove.840B.300d.txt'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "please load Glove Embeddings from http://nlp.stanford.edu/data/glove.840B.300d.zip and store in data/embeddings/")

        elif self.name == 'Glove-Wiki-50':
            self.path = '../data/embeddings/glove.6B.50d.txt'
            if not os.path.isfile(self.path):
                self.path = 'data/embeddings/glove.6B.50d.txt'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "please load Glove Embeddings from http://nlp.stanford.edu/data/glove.6B.zip and store in data/embeddings/")

        elif self.name == 'Glove-Wiki-100':
            self.path = '../data/embeddings/glove.6B.100d.txt'
            if not os.path.isfile(self.path):
                self.path = 'data/embeddings/glove.6B.100d.txt'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "please load Glove Embeddings from http://nlp.stanford.edu/data/glove.6B.zip and store in data/embeddings/")

        elif self.name == 'Glove-Wiki-200':
            self.path = '../data/embeddings/glove.6B.200d.txt'
            if not os.path.isfile(self.path):
                self.path = 'data/embeddings/glove.6B.200d.txt'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "please load Glove Embeddings from http://nlp.stanford.edu/data/glove.6B.zip and store in data/embeddings/")

        elif self.name == 'Glove-Wiki-300':
            self.path = '../data/embeddings/glove.6B.300d.txt'
            if not os.path.isfile(self.path):
                self.path = 'data/embeddings/glove.6B.300d.txt'
                if not os.path.isfile(self.path):
                    raise Exception(
                        "please load Glove Embeddings from http://nlp.stanford.edu/data/glove.6B.zip and store in data/embeddings/")


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




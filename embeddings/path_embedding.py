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

class PathEmbeddings(Embeddings):

    def __init__(self, params):
        """
        This class calls the FastText data
        """

        # Load the super class
        super(PathEmbeddings, self).__init__()

        if 'path' in params:
            self.path = params['path']
            if 'name' not in params:
                self.name = params['path']
            else:
                self.name = params['name']
        else:
            raise Exception("No path set")

        self.header_lines = -1
        if 'header_lines' in params:
            self.header_lines = params['header_lines']


        if not os.path.isfile(self.path):
            raise Exception(
                "no file found")


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
                if k <= self.header_lines: continue
                term, embedding = self.parse_line(line)
                self.add_term(term, preload=preload)
                embeddings.append(embedding)
        special_embeddings = self.add_special_embeddings(len(embeddings[0]), preload=preload)
        if special_embeddings != []:
            embeddings = embeddings + special_embeddings

        return np.array(embeddings)

    def get_name(self):
        return self.name




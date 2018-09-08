import os.path
import numpy as np
from embeddings import Embeddings

"""
THIS IS A CHILD OF EMBEDDINGS!

THIS SCRIPT SHOULD BE USED TO INITIALIZE A NEW EMBEDDING DATASET

"""




class DUMMYEmbeddings(Embeddings):

    def __init__(self):
        """
        This class calls the FastText data
        """

        # Load the super class
        super(DUMMYEmbeddings, self).__init__()

        # Name of the embedding dataset
        self.name = 'DUMMYNAME'

        ################################################################################################################
        # Here we define the path to the embeddings
        ################################################################################################################

    def parse_line(self, line):
        """
        Here we parse the  line of the embedding file and return the term and the embeddings
        :param line:
        :return:
        """
        pass
        # elems = line.split()
        # term = elems[0]
        # embedding = [float(i) for i in elems[1:]]
        # return term, embedding


    def load_top_k(self, K):
        """
        Option for loading strategy: Only load top k of embeddings assuming that they are ordered in frequency
        :param K: Number of top k embeddings to be retrieved
        :return:embeddings matrix as numpy
        """
        pass
        # embeddings = []
        # with open(self.path, 'r') as f:
        #     for k, line in enumerate(f):
        #         if k == 0: continue
        #         if k > K: break
        #         term, embedding = self.parse_line(line)
        #         self.add_term(term)
        #         embeddings.append(embedding)
        #
        # return embeddings

    def get_name(self):
        return self.name

    # def add_special_embeddings(self, embedding_size):
    #
    #     PAD = '<PAD>'
    #     START = '<S>'
    #     END = '</S>'
    #     UNK = '<UNK>'
    #
    #     embeddings = []
    #
    #     if self.get_start_token() is None:
    #         self.add_term(START)
    #         embeddings.append(np.random.random(embedding_size))
    #     if self.get_pad_token() is None:
    #         self.add_term(PAD)
    #         embeddings.append(np.random.random(embedding_size))
    #     if self.get_end_token() is None:
    #         self.add_term(END)
    #         embeddings.append(np.random.random(embedding_size))
    #     if self.get_unk_token() is None:
    #         self.add_term(UNK)
    #         embeddings.append(np.random.random(embedding_size))
    #
    #     return embeddings

    def load_in_dict(self):
        """
        Option for loading strategy: Only load the embeddings for words in the dictionary of the dataset. This
        assumes that the dictionary has already been defined
        :return: embeddings matrix as numpy
        """
        pass

        # # List of embeddings that are in dictionary
        # embeddings = []
        #
        # # position the embeddings are in this list (embeddings) and where they are in the true dataset. This is
        # # necessary for post reordering
        # positions = []
        #
        # # number of words in the vocabulary to identify if we have already found all embeddings
        # vocab_size = self.get_vocab_size()
        #
        # # Loop through embeddings
        # with open(self.path, 'r') as f:
        #     for k, line in enumerate(f):
        #
        #         # Skip first line
        #         if k == 0: continue
        #
        #         # parse the line to identify which word this embedding is for
        #         term, embedding = self.parse_line(line)
        #
        #         # get the position of the word in the dictionary. This returns None if the word is not in dictionary
        #         position = self.get_position(term)
        #
        #         # skip if word is not in dictionary
        #         if not position: continue
        #
        #         # if found, add it to the embedding list
        #         embeddings.append(embedding)
        #
        #         # store the current position of the embedding, and the position it should be in according to the vocab
        #         positions.append([len(positions), position])
        #
        #         # If we have found all embeddings we are done
        #         if len(positions) == vocab_size:
        #             break
        #
        # # Resort the embeddings according to the real order as defined by the precomputed vocabulary
        # positions = np.array(positions)
        # embeddings = np.array(embeddings)
        # resorted_positions = positions[positions[:,1].argsort()]
        # vocab_positions = resorted_positions[:,0]
        # embeddings = embeddings[vocab_positions]
        #
        # return embeddings



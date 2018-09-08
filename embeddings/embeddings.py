from data_loading.data_utils import tokenize, pickle_call, pickle_dump
import torch.nn as nn
import torch
import pdb
import numpy as np

"""
SUPER CLASS OF EMBEDDING DATASETS
"""

class Embeddings(object):

    def __init__(self,  vocab_list=None, embedding_size=300):
        """
        Initialize the vocabulary dictionary and list
        :param vocab_list:
        :param embedding_size:
        :param load_directory:
        """

        # current size of the dictionary
        self.vocab_size = 0

        # to-be embeddings size, however not necessary if intialized from another source
        self.embedding_size = embedding_size

        # vocabulary list and dict. If list is given we use it
        # list: fast term retrieval based on position
        # dict: fast position retrieval based on term

        if vocab_list is None:
            self.vocab_list = []
            self.vocab_dict = {}
            self.preload_vocab_list = []
            self.preload_vocab_dict = {}
            self.preload_vocab_size = 0
            self.preload_vocab_count = []
        else:
            self.build_vocab_dict(vocab_list)

        # Embeddings have not yet been set.
        self.embeddings = None

    def build_vocab_dict(self, vocab_list):
        """
        If a list of vocabulary is given we define the dictionary based on this
        :param vocab_list:
        :return:
        """
        self.vocab_list = vocab_list
        self.vocab_dict = {}
        for p, term in enumerate(vocab_list):
            self.add_term(term, position=p)

    def add_term(self, term, position=None, preload=False):
        """
        For sequential adding of terms to the dicitonary
        :param term:
        :param position: if position is given we assume that its already in the list and only add it to the dictionary
        :return:
        """

        # Set the variables to the preloading global variables if we are currently only preloading the vocab
        if preload:
            vocab_size= self.preload_vocab_size
            vocab_dict = self.preload_vocab_dict
            vocab_list = self.preload_vocab_list
        else:
            vocab_size= self.vocab_size
            vocab_dict = self.vocab_dict
            vocab_list = self.vocab_list



        if self.embeddings is not None:
            raise Exception("Embedding matrix has already been initialized")
        if term not in vocab_dict:
            vocab_size += 1
            if position is None:
                vocab_dict[term] = len(vocab_dict)
                vocab_list.append(term)
                if preload:
                    self.preload_vocab_count.append([term,0])
            else:
                vocab_dict[term] = position

    def preload_embeddings(self):
        self.preloaded_embeddings = self.load_top_k(K=float('inf'), preload=True)

    def load_in_dict(self, K, preload):
        embeddings_list = []
        for term in self.vocab_list:
            embeddings_list.append(self.preloaded_embeddings[self.preload_vocab_dict[term]])
        return np.array(embeddings_list)

    def initialize_embeddings(self, embedding_func=None, K=float('inf'), preload=False):
        """
        Intialize the embeddings based on the initialization funciton and the vocabulary
        :param embedding_func: if a funciton is given we load embeddings
        :param parameters: dictionary of parameters e.g. {k:10} for embedding initialization
        :return:
        """
        # If the embeddings have not yet been defined we intialze them
        if self.embeddings is None:

            # If no embedding funciton is set, we initialize them based on the the default pytorch settings
            if not embedding_func:
                self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)

            # Retrieve embeddings
            embeddings = embedding_func(K, preload)
            # redefine the embedding size
            self.embedding_size = len(embeddings[0])

            # initialize and load embeddings
            self.embeddings = nn.Embedding(len(embeddings), self.embedding_size, padding_idx=self.get_pad_pos())
            self.embeddings.weight = nn.Parameter(torch.from_numpy(embeddings))


    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab_list(self):
        return self.vocab_list

    def get_vocab_dict(self):
        return self.vocab_dict

    def get_embeddings(self):
        if self.embeddings is None:
            raise Exception("Embeddings have not been initialized")
        return self.embeddings

    def add_special_embeddings(self, embedding_size, preload=False):
        """
        The FastText data set does not include pretrained embeddings for PAD, START, END and UNK tokens so we need to
        initialize them randomly here.
        :param embedding_size:
        :return:
        """

        embeddings = []

        # If we have preloaded the embeddings we need to add the special embeddings to both the preloaded vocabulary
        # and the normal vocab.
        # Logic is: first check if it does not yet exist and if not add it.
        if preload:
            PAD = self.get_pad_token(initialize=False, preloaded_dict=True)
            if PAD is None:
                PAD = self.get_pad_token(initialize=True, preloaded_dict=True)
                embeddings.append(np.random.randn(embedding_size))
            self.add_term(PAD, preload=False)

            UNK = self.get_unk_token(initialize=False, preloaded_dict=True)
            if UNK is None:
                UNK = self.get_unk_token(initialize=True, preloaded_dict=True)
                embeddings.append(np.random.randn(embedding_size))
            self.add_term(UNK, preload=False)

            START = self.get_start_token(initialize=False, preloaded_dict=True)
            if START is None:
                START = self.get_start_token(initialize=True, preloaded_dict=True)
                embeddings.append(np.random.randn(embedding_size))
            self.add_term(START, preload=False)

            END = self.get_end_token(initialize=False, preloaded_dict=True)
            if END is None:
                END = self.get_end_token(initialize=True, preloaded_dict=True)
                embeddings.append(np.random.randn(embedding_size))
            self.add_term(END, preload=False)

        # If we do not preload we just add the special tokens normally
        else:
            PAD = self.get_pad_token(initialize=False, preloaded_dict=False)
            if PAD is None:
                self.get_pad_token(initialize=True, preloaded_dict=False)
                embeddings.append(np.random.randn(embedding_size))

            UNK = self.get_unk_token(initialize=False, preloaded_dict=False)
            if UNK is None:
                self.get_unk_token(initialize=True, preloaded_dict=False)
                embeddings.append(np.random.randn(embedding_size))

            START = self.get_start_token(initialize=False, preloaded_dict=False)
            if START is None:
                self.get_start_token(initialize=True, preloaded_dict=False)
                embeddings.append(np.random.randn(embedding_size))

            END = self.get_end_token(initialize=False, preloaded_dict=False)
            if END is None:
                self.get_end_token(initialize=True, preloaded_dict=False)
                embeddings.append(np.random.randn(embedding_size))

        return embeddings

    def get_position(self, term, initialize=False, count_up=False, return_UNK=True, preloaded_dict=False):
        """
        returns the position of a tern, if initialize is set to True we initialize it and return the new position
        :param term:
        :param initialize: True, False if the term should be added to the vocab
        :param count_up: boolean if we want to count up the occurency of terms
        :param return_UNK: if UNK position should be returned if the term is not in vocab
        :param preloaded_dict: if instead of the normal vocab dict, the preloaded_dict should be checked
        :return:
        """


        # select the correct vocab dict based on setting of preloaded_dict
        if preloaded_dict:
            vocab_dict = self.preload_vocab_dict
        else:
            vocab_dict = self.vocab_dict

        # This is a special case. therefore we "overwrite" the top selection. counting up is only necessary if we
        # have already preloaded the embeddings. In that case we want to count up the occurence in preload_vocab_dict
        if count_up and term in self.preload_vocab_dict:

            # Count the term up
            self.preload_vocab_count[self.preload_vocab_dict[term]][1] += 1

            # if the term is not yet in our vocab, add it
            if term not in self.vocab_dict:
                self.add_term(term, preload=False)

        # if we do not want to initialize the term and its not in our vocab
        if not initialize and term not in vocab_dict:
            # and if return_UNK is set to true, we return the UNK position
            if return_UNK:
                return self.get_unk_pos()
            # Else we return None
            else:
                return None

        # we only come here if initialize is set to True, therefore we add the term to our vocab if its not yet added
        # preload=preloaded_dict works because both rely on the same boolean logic
        if term not in vocab_dict:
            self.add_term(term, preload=preloaded_dict)

        # we return the positon of the term
        return vocab_dict[term]

    def encode_sentence_array(self, term_list, initialize=False, count_up=False):
        """
        Given an array of terms we iteratively retrieve the positions of the embeddings and add them to vocab if
        initialize is set to True
        :param term_list: array of terms
        :param initialize:
        :return:
        """
        positions = []
        for term in term_list:
            position = self.get_position(term, initialize=initialize, count_up=count_up)
            if position is None:
                position = self.get_unk_pos(initialize=initialize)
            if position is not None:
                positions.append(position)
        return positions

    def encode_sentence(self, sentence, initialize=False, count_up=False):
        """
        First tokenize the sentence and then call encode_sentence_array
        :param sentence: string of sentence
        :param initialize:
        :return:
        """
        term_list = tokenize(sentence, start_token=self.get_start_token(), end_token=self.get_end_token())
        return self.encode_sentence_array(term_list, initialize=initialize, count_up=count_up)

    def decode_position(self, position):
        """
        given a position retrieve associated term
        :param position:
        :return:
        """
        if position >= self.vocab_size:
            raise Exception("Out of vocab")
        return self.vocab_list[position]

    def decode_position_list(self, position_list):
        """
        given a list of position reformat it into a sentence string
        :param position_list:
        :return:
        """
        sentence = ""
        for position in position_list:
            sentence += self.decode_position(position) + " "
        return sentence

    def load_top_k(self, K, preload=False):
        """
        Defined in child classes
        :param K:
        :return:
        """
        pass

    def dump(self, file_name):
        """
        Store everything in a dict and dump it to pickle file
        :param file_name: path and filename where it should be stored
        :return:
        """
        dump_dict = {}
        dump_dict['vocab_size'] = self.vocab_size
        dump_dict['vocab_list'] = self.vocab_list
        dump_dict['vocab_dict'] = self.vocab_dict
        dump_dict['embeddings'] = self.embeddings

        pickle_dump(file_name, dump_dict)

    def load(self, file_name):
        """
        load all data from file if we have it
        :param file_name: path and filename where it should be stored
        :return:
        """
        dump_dict = pickle_call(file_name)
        if dump_dict is not None:
            self.vocab_size = dump_dict['vocab_size']
            self.vocab_list = dump_dict['vocab_list']
            self.vocab_dict = dump_dict['vocab_dict']
            self.embeddings = dump_dict['embeddings']
            return True
        return False

    def get_unk_pos(self, initialize=True):
        """
        Returns the position of the UNK token
        :return:
        """
        UNK = self.get_position('<UNK>', initialize=initialize)
        if UNK is not None: return UNK
        UNK = self.get_position('<unk>', initialize=initialize)
        if UNK is not None: return UNK

    def get_pad_pos(self, initialize=True):
        """
        Returns the position of the PAD token
        :return:
        """
        PAD = self.get_position('<PAD>', initialize=initialize)
        if PAD is not None: return PAD
        PAD = self.get_position('<pad>', initialize=initialize)
        if PAD is not None: return PAD

    def get_start_pos(self,initialize=True):
        """
        Returns the position of the START token
        :return:
        """
        START = self.get_position('<S>', initialize=initialize)
        if START is not None: return START
        START = self.get_position('<s>', initialize=initialize)
        if START is not None: return START

    def get_end_pos(self, initialize=True):
        """
        Returns the position of the END token
        :return:
        """
        END = self.get_position('</S>', initialize=initialize)
        if END is not None: return END
        END = self.get_position('</s>', initialize=initialize)
        if END is not None: return END

    def get_start_token(self, initialize=True, preloaded_dict=False):
        """
        Returns the string of the START token
        :return:
        """
        start_token = '<S>'
        if self.get_position(start_token, initialize=initialize, return_UNK=False, preloaded_dict=preloaded_dict) is not None:
            return start_token

        start_token = '<s>'
        if self.get_position(start_token, initialize=initialize, return_UNK=False, preloaded_dict=preloaded_dict) is not None:
            return start_token

    def get_end_token(self, initialize=True, preloaded_dict=False):
        """
        Returns the string of the END token
        :return:
        """
        end_token = '</S>'
        if self.get_position(end_token, initialize=initialize, return_UNK=False, preloaded_dict=preloaded_dict) is not None:
            return end_token

        end_token = '</s>'
        if self.get_position(end_token, initialize=initialize, return_UNK=False, preloaded_dict=preloaded_dict) is not None:
            return end_token

    def get_unk_token(self, initialize=True, preloaded_dict=False):
        """
        Returns the string of the UNK token
        :return:
        """
        unk_token = '<UNK>'
        if self.get_position(unk_token, initialize=initialize, return_UNK=False, preloaded_dict=preloaded_dict) is not None:
            return unk_token

        unk_token = '<unk>'
        if self.get_position(unk_token, initialize=initialize, return_UNK=False, preloaded_dict=preloaded_dict) is not None:
            return unk_token

    def get_pad_token(self, initialize=True, preloaded_dict=False):
        """
        Returns the string of the PAD token
        :return:
        """
        pad_token = '<PAD>'
        if self.get_position(pad_token, initialize=initialize, return_UNK=False, preloaded_dict=preloaded_dict) is not None:
            return pad_token

        pad_token = '<pad>'
        if self.get_position(pad_token, initialize=initialize, return_UNK=False, preloaded_dict=preloaded_dict) is not None:
            return pad_token


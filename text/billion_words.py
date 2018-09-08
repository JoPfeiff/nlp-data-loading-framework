from text.text_data import TextData
from data_loading.data_utils import pad_positions
import os
from tqdm import *

class BillionWordsData(TextData):
    """
    This class is a child of TextData
    We load the billion word benchmark data set and store it in self.data_set
    Because of the size of this data set we do not store the raw data, but only the positions
    """
    def __init__(self, embeddings, data_params):

        super(BillionWordsData, self).__init__( 'BillionWords', embeddings)

        # Initialize the two dicts for train and dev buckets
        self.data_sets['train_buckets'] = {}
        self.data_sets['dev_buckets'] = {}

        # TODO: Check the right bucket sizes
        self.buckets = [10,20,30,50,100, 200]

        # For testing purposes or to generate a smaller data set we can define the number of k files that we want to
        # load
        if 'k' in data_params:
            self.k = data_params['k']
        else:
            self.k = float('inf')

        # Define all variables for each bucket
        for bucket in self.buckets:
            # data_set + "_buckets"
            self.data_sets['train_buckets'][bucket] = {}
            self.data_sets['dev_buckets'][bucket] = {}
            self.data_sets['train_buckets'][bucket]['data'] = []
            self.data_sets['dev_buckets'][bucket]['data'] = []
            self.data_sets['train_buckets'][bucket]['bucket_size'] = bucket
            self.data_sets['dev_buckets'][bucket]['bucket_size'] = bucket

            self.data_sets['train_buckets'][bucket]['length'] = 0
            self.data_sets['dev_buckets'][bucket]['length'] = 0
            self.data_sets['train_buckets'][bucket]['position'] = 0
            self.data_sets['dev_buckets'][bucket]['position'] = 0

            self.data_sets['train_buckets'][bucket]['buckets'] = bucket
            self.data_sets['dev_buckets'][bucket]['buckets'] = bucket


    def load_file(self, data_type, file_name, initialize_term=False):
        """
        Here we just load one of the many files of the billion word benchmark data set
        :param data_type: 'train' or 'dev'
        :param file_name: the name of the file
        :param initialize_term: if we are going
        :return:
        """
        print("Parsing file " + file_name)
        PAD_position = self.embeddings.get_pad_pos(initialize=True)

        with open(file_name, 'r') as file:
            for i, line in enumerate(tqdm(file)):

                #TODO delete only for testing
                # if i == 10000: break

                sentence = {}

                # if self.embeddings is not None:
                sentence_positions = self.embeddings.encode_sentence(line.decode('utf-8'), initialize=initialize_term, count_up=True)
                sentence['sentence_positions'] = sentence_positions
                sent_length = len(sentence_positions)
                sentence['length'] = sent_length
                for bucket in self.buckets:
                    if sent_length <= bucket:
                        sentence['sentence_positions'] = pad_positions(sentence['sentence_positions'], PAD_position, bucket)
                        self.data_sets[data_type + '_buckets'][bucket]['data'].append(sentence)
                        self.data_sets[data_type + '_buckets'][bucket]['length'] += 1
                        break

    def load_billion_words(self, data_set='train', directory=None, initialize_term=False):

        if data_set  in self.data_sets:
            return self.data_sets[data_set+'_buckets']

        if directory is None:
            dir = '../data/'
            if not os.path.isdir(dir):
                dir = 'data/'
                if not os.path.isdir(dir):
                    raise Exception('no data directory found')

        if data_set == 'train':
            left = '1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-000'
            right = '-of-00100'
            if self.k > 100:
                self.k = 100
        elif data_set == 'dev':
            left = '1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en.heldout-000'
            right = '-of-00050'
            if self.k > 50:
                self.k = 50

        else:
            raise Exception("please set data_set to either 'train' or 'dev' ")

        for i in range(self.k):
            if i < 10:
                middle = '0' + str(i)
            else:
                middle = str(i)

            path = dir + left + middle + right

            if not os.path.isfile(path):
                print("file " + path + " not found" )
                continue
            self.load_file(data_set, path, initialize_term=initialize_term)

        return self.data_sets[data_set + '_buckets']

    def bucketize_data(self, data_set, initialize):
        print("Billion Words Data already bucketized")


#
# if __name__ == '__main__':
#
#     bwd = BillionWordsData()
#
#     file_name = '../data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00000-of-00100'
#
#     bwd.load_billion_words(data_set='train', k=1)
#
#     print("done")




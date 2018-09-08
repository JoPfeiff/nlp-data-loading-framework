import json
import os.path
from text.text_data import TextData
from data_loading.data_utils import pad_positions


class SNLIData(TextData):
    """
    Class that loads and shuffles the SNLI Dataset
    please load SNLI data from https://nlp.stanford.edu/projects/snli/snli_1.0.zip and store it in /data/
    """

    def __init__(self,label_dict, data_params=None, bucket_params=None, embeddings=None):

        """
        Initialize the class using parameters e.g. which data from SNLI should be stored. SNLI also needs buckets
        These are initialized by default if not specified.
        :param label_dict: dictionary of labels with a corresponding integer. e.g. {'neutral':0, 'entailment': 1}
        :param data_params: set of parameters that should be called see bottom
        :param bucket_params: For SNLI buckets for sentence1 and sentence2 need to be defined
        """
        super(SNLIData, self).__init__('SNLI', embeddings)

        # Default parameters to be called from SNLI
        if data_params is None or len(data_params) == 0:
            self.data_params = {
                  "annotator_labels"          : False,
                  "captionID"                 : False,
                  "gold_label"                : True,
                  "paidID"                    : False,
                  "sentence1"                 : True,
                  "sentence1_binary_parse"    : False,
                  "sentence1_parse"           : False,
                  "sentence2"                 : True,
                  "sentence2_binary_parse"    : False,
                  "sentence2_parse"           : False}
        else:
            self.data_params = data_params

        # label dict  e.g. {'neutral':0, 'entailment': 1}
        self.label_dict = label_dict

        # Default buckets
        if bucket_params is None:
            self.bucket_params = [
                        [10,10],
                        [10,20],
                        [15,10],
                        [15,20],
                        [20,10],
                        [20,20],
                        [30,10],
                        [30,20],
                        [40,20],
                        [100,100]
                        ]
        else:
            self.bucket_params = bucket_params

    def load_snli(self, data_type='train', initialize_term=False):
        """
        This function calls the data set from file and stores it based on the predefined parameters
        :param data_type: 'train', 'test' or 'dev'
        :return: the list of dict elements (but also stored in self.data_set)
        """
        if data_type  in self.data_sets:
            return self.data_sets[data_type]

        # get paths for training, testing and dev data
        path = '../data/snli_1.0/snli_1.0_' + data_type +'.jsonl'
        if not os.path.isfile(path):
            path = 'data/snli_1.0/snli_1.0_'+ data_type +'.jsonl'
            if not os.path.isfile(path):
                raise Exception("please load SNLI data from https://nlp.stanford.edu/projects/snli/snli_1.0.zip and store it in /data/")

        # set up a list of dict elements that we want to load from the dataset
        param_values = []
        for key, value in self.data_params.items():
            if value:
                param_values.append(key)

        # load only the json values that we want and append it to our dataset
        data = []
        with open(path) as f:
            for i, line in enumerate(f):
                elem = {}
                j = json.loads(line)
                for param in param_values:
                    elem[param] = j[param]
                    if param == 'gold_label' and self.label_dict is not None:
                        elem['label'] = self.label_dict[j[param]]

                # If an embedding object has been passed we can encode the sentences with the corresponding positions
                if 'sentence1' in param_values and self.embeddings is not None:
                    elem['sentence1_positions'] = self.embeddings.encode_sentence(elem['sentence1'], initialize=initialize_term, count_up=True)
                    elem['sentence1_length'] = len(elem['sentence1_positions'] )
                if 'sentence2' in param_values and self.embeddings is not None:
                    elem['sentence2_positions'] = self.embeddings.encode_sentence(elem['sentence2'], initialize=initialize_term, count_up=True)
                    elem['sentence2_length'] = len(elem['sentence2_positions'])
                data.append(elem)

        self.data_sets[data_type] = data

        return data


    def bucketize_data(self, data_set, initialize):
        """
        stores the data points in the designated buckets and stores meta data
        a bucket is defined by the maximum length of sentence1 and sentence2 respectively
        :param data_set: 'train' 'test' 'dev'
        :return:
        """

        PAD_position = self.embeddings.get_pad_pos(initialize=True)

        bucket_name = data_set + "_buckets"

        if bucket_name in  self.data_sets:
            return None

        # dictionary in which the data of the different buckets will be stored
        bucketized = {}

        # define metadata for each bucket
        for b1, b2 in self.bucket_params:
            bucketized[str(b1) + '_' + str(b2)] = {}
            # list of data points
            bucketized[str(b1) + '_' + str(b2)]['data'] = []
            # max lengths of sentence1 and sentence2 respectively
            bucketized[str(b1) + '_' + str(b2)]['buckets'] = [b1, b2]
            # nr of data points in the bucket (will be counted up)
            bucketized[str(b1) + '_' + str(b2)]['length'] = 0
            # position of sampled data (will be shuffled first and then iteratively retrieved)
            bucketized[str(b1) + '_' + str(b2)]['position'] = 0

        # retrieve defined data_set ('train', 'test', 'dev')
        data = self.data_sets[data_set]

        # loop through elements of data set, store the data point in the corresponding bucket and count up the length
        for elem in data:
            len1 = elem['sentence1_length']
            len2 = elem['sentence2_length']

            for b1, b2 in self.bucket_params:
                if len1 <= b1 and len2 <= b2:
                    elem['sentence1_positions'] = pad_positions(elem['sentence1_positions'], PAD_position, b1)
                    elem['sentence2_positions'] = pad_positions(elem['sentence2_positions'], PAD_position, b2)
                    bucketized[str(b1) + '_' + str(b2)]['data'].append(elem)
                    bucketized[str(b1) + '_' + str(b2)]['length'] += 1
                    break

        # store the bucketized data in the class dictionary
        self.data_sets[bucket_name] = bucketized


if __name__ == '__main__':
    labels = {'neutral': 0, 'entailment': 1, 'contradiction': 2, '-': 3}
    data = SNLIData(labels)

    gen = data.generator('train', 64)

    nr_data_points = 0


from text.SNLI_data_loading import SNLIData
from text.billion_words import BillionWordsData
from embeddings.fasttext_embeddings import FastTextEmbeddings, FASTTEXT_NAMES
from embeddings.polyglot_embeddings import PolyglotEmbeddings, POLYGLOT_NAMES
from embeddings.lear_embeddings import LearEmbeddings, LEAR_NAMES
from embeddings.glove_embeddings import GloveEmbeddings, GLOVE_NAMES
from embeddings.path_embedding import PathEmbeddings
import os


class DataLoader:
    """
    This class loads the data sets for the different classes.
    The data set should always consist of a list of dict elements each consisting all data relevant for a single
    data point.
    """
    
    def __init__(self, data_set='SNLI', embeddings_initial='random', embedding_loading='in_dict',
                 embedding_params={}, param_dict={}, K_embeddings=float('inf'), bucket_params=None, embedding_size=300, vocab_list=None ):
        """
        We initialize some parameters and define the dataset we want to call
        :param data_set: Which dataset should be called e.g. "SNLI"
        :param param_dict: What data should be loaded from file (JSON format)
        :param bucket_params: for SNLI we need to define buckets
        :param embeddings: 'random', 'FastText'
        """

        # If this is true, the data has been loaded from file
        self.loaded = False

        # Only load this amount of embeddings
        self.K_embeddings = K_embeddings

        # Store the embedding type: currently ['top_k', 'in_dict'] work
        self.embedding_loading = embedding_loading

        self.data_set = data_set
        # How we want to initialize the embeddings. If it is set to random we initialize the embeddings using the
        # default pytorch settings. The embedding_size is only needed for this setting because otherwise the
        # embedding size from the loaded dataset is fixed
        self.embeddings_initial = embeddings_initial
        if self.embeddings_initial == 'random':
            self.embedding_func = None
            self.embedding_params = None
            self.embedding = None

        # if something different to 'random' is defined, e.g. 'FastText', these embeddings are loaded
        else:

            # we store the dictionary of specified parameters for embedding loading
            self.embedding_params = embedding_params

            # We select the correct class that is a child of Embeddings to initialize the embeddings
            if self.embeddings_initial in FASTTEXT_NAMES:
                self.embedding = FastTextEmbeddings(self.embeddings_initial)
            elif self.embeddings_initial in POLYGLOT_NAMES:
                self.embedding = PolyglotEmbeddings()
            elif self.embeddings_initial in LEAR_NAMES:
                self.embedding = LearEmbeddings()
            elif self.embeddings_initial in GLOVE_NAMES:
                self.embedding = GloveEmbeddings(self.embeddings_initial)
            elif self.embeddings_initial == "Path":
                self.embedding = PathEmbeddings(self.embedding_params)
            else:
                raise Exception("No valid embedding was set")

            # Currently two selection strategies are defined for loading the embeddings
            # first is to load only those embeddings that are actually in our dictionary
            # this of course entails, that we have already loaded our dictionary
            if embedding_loading == 'in_dict':
                self.embedding_func = self.embedding.load_in_dict

            # Or we load the top_k of our embeddings. This assumes that the embeddings are stored in order of
            # frequency. k has to be defined in embedding_params
            elif embedding_loading == 'top_k':
                self.embedding_func = self.embedding.load_top_k

        # Loading the data for the NLI task
        if data_set == "SNLI":
            self.labels = {'neutral': 0, 'entailment': 1, 'contradiction': 2, '-': 3}
            self.data_ob = SNLIData(self.labels, data_params=param_dict, bucket_params=bucket_params,
                                    embeddings=self.embedding)
            self.load_class_data = self.data_ob.load_snli
            self.generator = self.data_ob.generator
        elif data_set == 'billion_words':
            self.data_ob = BillionWordsData(embeddings=self.embedding, data_params=param_dict)
            self.load_class_data = self.data_ob.load_billion_words
            self.generator = self.data_ob.generator
        else:
            raise Exception("No valid data_set set was set")

        # The name is derived from all chosen parameters and dataset names. This is necessary to define a distinct
        # pickle file name
        self.name = data_set + "_" + self.embedding.name + "_" + self.embedding_loading +  "_" + str(embedding_size)

    def initialize_embeddings(self):
        """
        This function initializes the embeddings as defined in the __init__()
        :return:
        """
        self.embedding.initialize_embeddings(self.embedding_func, K=self.K_embeddings)

    def load_in_dict(self, K=None):
        """
        This function combines both the classifcation dataset and the embedding dataset.
        The logic is
            1.      preload all the embeddings
            2.      go through the classification data set and check which embedded terms actually occur and define positions
            3.      select the subset of embeddings that is relevant
            (3.)    if K is defined we loop through the classifcation data set once more and redefine the positoins in
                    the classifcation data set
        :param K:
        :return:
        """

        # We first preload all the embeddings to know for which terms we actually have embeddings
        self.embedding.preload_embeddings()

        # We then call all the classification data
        self.get_train_data(initialize_term=False)
        self.get_test_data(initialize_term=False)
        self.get_dev_data(initialize_term=False)

    def dump(self, path='data/pickles/'):
        """
        We dump the both data and embeddings using a predefined self.name to be able to recover it in the future
        :param path:
        :return:
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self.data_ob.dump(path + self.name + '_data.pkl')
        self.embedding.dump(path + self.name + '_embeddings.pkl')

    def load(self, path='data/pickles/'):
        """
        We retrieve the stored data if it  exists
        :param path:
        :return:
        """
        class_data_loaded = self.data_ob.load(path + self.name + '_data.pkl')
        embedding_data_loaded = self.embedding.load(path + self.name + '_embeddings.pkl')
        if class_data_loaded and embedding_data_loaded:
            self.loaded = True



    def get_train_data(self, initialize_term=False):
        # if self.data_set == "SNLI":
        return self.load_class_data('train', initialize_term=initialize_term)
        # else:
        #     raise Exception("Dataset not specified")


    def get_test_data(self, initialize_term=False):
        # if self.data_set == "SNLI":
        if self.data_set == "billion_words":
            print('test data does not exist')
            return None

        return self.load_class_data('test', initialize_term=initialize_term)
        # else:
        #     raise Exception("Dataset not specified")


    def get_dev_data(self, initialize_term=False):
        # if self.data_set == "SNLI":
        return self.load_class_data('dev', initialize_term=initialize_term)
        # else:
        #     raise Exception("Dataset not specified")

    def get_labels(self):
        if self.data_set == "billion_words":
            print('labels do not exist')
            return None
        return self.labels

    def get_all_and_dump(self,path='data/pickles/'):
        """
        Generate all the data by default and pickle it.
        :param path: where the pickles should be stored
        :return:
        """

        # If the data has already been loaded, break
        if self.loaded: return None

        # initialize_term = False
        if self.embedding_loading == 'top_k':
            initialize_term = False
            self.initialize_embeddings()
            print("Loading train data..")
            self.get_train_data(initialize_term=initialize_term)
            print("Loading test data..")
            self.get_test_data(initialize_term=initialize_term)
            print("Loading dev data..")
            self.get_dev_data(initialize_term=initialize_term)
            self.get_labels()
        elif self.embedding_loading == 'in_dict':
            print("Initialising embeddings..")
            initialize_term = False
            self.load_in_dict()
            self.initialize_embeddings()
        print("Bucketizing train data..")
        self.data_ob.bucketize_data(data_set='train', initialize=initialize_term)
        print("Bucketizing test data..")
        self.data_ob.bucketize_data(data_set='test',initialize=initialize_term)
        print("Bucketizing dev data..")
        self.data_ob.bucketize_data(data_set='dev',initialize=initialize_term)
        print("Dumping all the data..")
        self.dump(path)

    def get_generator(self, data_set='train', batch_size=64, drop_last=True, initialize = True):
        """
        Retrieves a generator that yields shuffled batches. The SNLI dataset additionally includes the bucket sizes
        :param data_set: 'train', 'test', 'dev'
        :param batch_size: default 64
        :return: generator that you iterate through with generator.next()
        """
        return self.generator(data_set, batch_size, drop_last, initialize=initialize)


if __name__ == '__main__':

    embedding_params = {'path':'../data/embeddings/bow2.words', 'name':'bow2'}
    class_params = {'k':1}
    dl = DataLoader(embeddings_initial='Polyglot', embedding_loading='in_dict', embedding_params=embedding_params)
    # dl = DataLoader(data_set='billion_words',embeddings_initial='Polyglot', embedding_loading='in_dict', embedding_params=embedding_params, param_dict=class_params)
    dl.load('../data/pickles/')
    dl.get_all_and_dump('../data/pickles/')

    gen = dl.get_generator(drop_last=False, initialize=True)
    # tr = dl.get_train_data()
    nr_data_points = 0

    # short analysis if the amount of data yielded equals the total amount of data points in the training set.
    # TLDC; yes it does
    while True:
        data , batch = gen.next()
        if data is None:
            break
        nr_data_points += len(data)

    print nr_data_points
    # print len(tr)







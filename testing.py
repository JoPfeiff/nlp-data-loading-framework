import unittest
from data_loading.data_loader import DataLoader
from data_loading.data_utils import pickle_call
import numpy as np

class DataLoadingTest(unittest.TestCase):


    def test_generator_data_length_fast_text_SNLI_in_dict(self):
        dl = DataLoader(embeddings_initial='FastText-Crawl', embedding_loading='in_dict')
        # dl = DataLoader(embeddings_initial='FastText', embedding_loading='load_dict',
        #                 embedding_params={'first_time_emb_load': False})
        dl.load('data/pickles/')
        dl.get_all_and_dump('data/pickles/')

        gen = dl.get_generator(drop_last=False, initialize=True)
        tr = dl.get_train_data()
        nr_data_points = 0

        # short analysis if the amount of data yielded equals the total amount of data points in the training set.
        # TLDC; yes it does
        while True:
            data, batch = gen.next()
            if data is None:
                break
            nr_data_points += len(data)

        self.assertEqual(nr_data_points, len(tr))

    def test_generator_data_length_Polyglot_SNLI_in_dict(self):
        dl = DataLoader(embeddings_initial='Polyglot', embedding_loading='in_dict')
        # dl = DataLoader(embeddings_initial='FastText', embedding_loading='load_dict',
        #                 embedding_params={'first_time_emb_load': False})
        dl.load('data/pickles/')
        dl.get_all_and_dump('data/pickles/')

        gen = dl.get_generator(drop_last=False, initialize=True)
        tr = dl.get_train_data()
        nr_data_points = 0

        # short analysis if the amount of data yielded equals the total amount of data points in the training set.
        # TLDC; yes it does
        while True:
            data, batch = gen.next()
            if data is None:
                break
            nr_data_points += len(data)

        self.assertEqual(nr_data_points, len(tr))

    def test_loaded_polyglot_embeddings(self):

        data = pickle_call('data/embeddings/polyglot-en.pkl')

        dl = DataLoader(embeddings_initial='Polyglot', embedding_loading='in_dict')
        dl.load('data/pickles/')
        dl.get_all_and_dump('data/pickles/')

        all_true = None

        for i in range(len(data[0])):

            term = data[0][i]
            embedding = data[1][i]

            if term in dl.embedding.vocab_dict:
                position = dl.embedding.vocab_dict[term]
                stored_embedding = dl.embedding.embeddings.weight[position].data.numpy()

                if all_true is None:
                    all_true = np.array_equal(embedding, stored_embedding)
                else:
                    all_true = all_true and np.array_equal(embedding, stored_embedding)

        self.assertTrue(all_true)

unittest.main()
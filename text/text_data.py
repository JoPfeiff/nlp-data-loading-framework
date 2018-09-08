from data_loading.data_utils import pickle_call, pickle_dump
import random


class TextData(object):
    """
    Class for all kinds of classification text data.
    An embeddings object needs to be passed that includes the positions of each word in the sentence.
    A generator function is added which is based on bucketized data which is also defined in the children of this class
    """

    def __init__(self, name, embeddings):
        """
        initialize the class with a name, the embedding object and an empty dict for the data_set
        :param name:
        :param embeddings:
        """
        self.name = name
        self.embeddings = embeddings
        self.data_sets = {}

    def get_name(self):
        return self.name

    def dump(self, file_name):
        pickle_dump(file_name, self.data_sets)

    def load(self, file_name):
        dump_dict = pickle_call(file_name)
        if dump_dict is not None:
            self.data_sets = dump_dict
            return True
        return False

    def generator(self, data_set, batch_size, drop_last=True, initialize = True):
        """
        This generator function should be called before EVERY epoch
        It loops through every data point exactly once
        Logic is:
            1. shuffle all buckets
            2. sample bucket based on nr. data points STILL in bucket
            3. yield the next *batch_size* data points from that bucket
        :param data_set:
        :param batch_size:
        :return:
        """

        PAD_position = self.embeddings.get_pad_pos(initialize=initialize)

        # # initialize data set if not yet done
        # if data_set not in self.data_sets:
        #     self.load_snli(data_set)
        #
        # # if data_set hasnt been bucketized yet, bucketize it
        # if data_set + "_buckets" not in self.data_sets:
        #     self.bucketize_data(data_set, initialize=initialize)

        # store all the bucket dictionary key values and bucket sizes
        bucket_names = []
        bucket_sizes = []

        # loop through each bucket and shuffle data set, store the key name and the nr. of data points in each bucket
        for key, bucket_dicts in self.data_sets[data_set + "_buckets"].items():
            random.shuffle(bucket_dicts['data'])
            bucket_names.append(key)
            bucket_sizes.append(bucket_dicts['length'])

        # number of data points in total not yet yielded
        nr_data_points = sum(bucket_sizes)

        # as long as we havent yielded each data point at least once, we continue sampling
        while nr_data_points > 0:

            # generate a random number in the range of the number of data points
            rand_int = random.randint(0,nr_data_points)

            # loop through the bucket sizes to identify which bucket was selected
            past_size = 0
            for bucket_id in range(len(bucket_sizes)):
                if rand_int <= bucket_sizes[bucket_id] + past_size and bucket_sizes[bucket_id] > 0:
                    bucket_name = bucket_names[bucket_id]
                    break
                past_size += bucket_sizes[bucket_id]

            # retrieve the sampled bucket
            current_bucket = self.data_sets[data_set + "_buckets"][bucket_name]

            # if not enough data points are left in the bucket, set the batch size to the number of data points left
            local_batch_size = min(current_bucket['length'] - current_bucket['position'], batch_size)


            # current position is the buckets' stored position of data points not yet yielded
            # upper_position is the position of the data point which will be yielded the next time this bucket will be
            # sampled from.
            upper_position = current_bucket['position'] + local_batch_size

            # we only yield incomplete batches if toss_incomplete is set to False
            if local_batch_size == batch_size or not drop_last:

                # we therefore yield all data points from the current bucket until the upper_position.
                # additionally the max sentence lenghts are yielded
                yield current_bucket['data'][current_bucket['position']: upper_position], current_bucket['buckets']

            # update the current position
            current_bucket['position'] = upper_position

            # decimate that number of data points left in that bucket
            bucket_sizes[bucket_id] -= local_batch_size

            # update the number of data points left
            nr_data_points = sum(bucket_sizes)

        # ensuring that the positions are reset for the next run
        for elem in self.data_sets[data_set + "_buckets"].items():
            elem[1]['position'] = 0
        yield None, None

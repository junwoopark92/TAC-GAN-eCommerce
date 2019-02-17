import os
import fire
import time
import sys
import traceback
from sklearn.externals import joblib
import random
import h5py
import numpy as np
from misc import get_logger, ges_Aonfig
from parse_metadata import EcommerceDataParser

from multiprocessing import Pool
from functools import partial
from six.moves import cPickle

def one_hot_encode_str_lbl(lbl, target, one_hot_targets):
    """
    Encodes a string label into one-hot encoding

    Example:
        input: "window"
        output: [0 0 0 0 0 0 1 0 0 0 0 0]
    the length would depend on the number of classes in the dataset. The
    above is just a random example.

    :param lbl: The string label
    :return: one-hot encoding
    """
    idx = target.index(lbl)
    return one_hot_targets[idx]


def get_one_hot_targets(target_file_path):
    target = []
    one_hot_targets = []
    n_target = 0
    try :
        with open(target_file_path) as f :
            target = f.readlines()
            target = [t.strip('\n') for t in target]
            n_target = len(target)
    except IOError :
        print('Could not load the labels.txt file in the dataset. A '
              'dataset folder is expected in the "data/datasets" '
              'directory with the name that has been passed as an '
              'argument to this method. This directory should contain a '
              'file called labels.txt which contains a list of labels and '
              'corresponding folders for the labels with the same name as '
              'the labels.')
        traceback.print_stack()

    lbl_idxs = np.arange(n_target)
    one_hot_targets = np.zeros((n_target, n_target))
    one_hot_targets[np.arange(n_target), lbl_idxs] = 1

    return target, one_hot_targets, n_target


class eCommerceData:
    def __init__(self, config):
        self.logger = get_logger()
        self.parse_data_path = config['PARSEMETA']['PARSE_DATA_PATH']
        self.category_path = config['PARSEMETA']['CATEGORY_PATH']
        self.n_log_print = config['PARSEMETA']['N_LOG_PRINT']
        self.doc_vec_size = config['PARSEMETA']['DOC_VEC_SIZE']
        self.train_dir_path = config['MAKEDB']['TRAIN_DIR_PATH']
        self.chunk_size = config['MAKEDB']['CHUNK_SIZE']
        self.temp_dir_path = config['MAKEDB']['TEMP_DIR_PATH']

        self.parser = EcommerceDataParser(config['PARSEMETA'], use=True)

    def get_train_indices(self, size, train_ratio):
        train_indices = np.random.rand(size) < train_ratio
        train_size = int(np.count_nonzero(train_indices))
        return train_indices, train_size

    def create_dataset(self, g, size, num_classes):
        g.create_dataset('docvec', (size, self.doc_vec_size), chunks=True, dtype=np.float32)
        g.create_dataset('cate', (size, num_classes), chunks=True, dtype=np.int32)
        g.create_dataset('asin', (size,), chunks=True, dtype='S14')

    def init_chunk(self, chunk_size, num_classes):
        chunk = {}
        chunk['docvec'] = np.zeros(shape=(chunk_size, self.doc_vec_size), dtype=np.float32)
        chunk['cate'] = np.zeros(shape=(chunk_size, num_classes), dtype=np.int32)
        chunk['asin'] = []
        chunk['num'] = 0
        return chunk

    def copy_chunk(self, dataset, chunk, offset):
        num = chunk['num']
        dataset['docvec'][offset:offset + num] = chunk['docvec'][:num]
        dataset['cate'][offset:offset + num] = chunk['cate'][:num]
        dataset['asin'][offset:offset + num] = chunk['asin'][:num]

    def save_caption_vectors_products(self, train_ratio=0.8):
        target, one_hot_targets, n_target = get_one_hot_targets(self.category_path)

        train_list = []
        with open(self.parse_data_path) as cap_f:
            for i, line in enumerate(cap_f):
                row = line.strip().split('\t')
                asin = row[0]
                categories = row[1]
                title = row[2]
                imgid = asin+'.jpg'
                onehot_cate = np.sum(np.asarray([one_hot_encode_str_lbl(class_name, target, one_hot_targets)
                                               for class_name in categories.split(',')]), axis=0).astype(bool).astype(int)

                train_list.append((imgid, onehot_cate, title))

                if i % self.n_log_print == 0:
                    print(i, train_list[-1])

        chunk_size = self.chunk_size
        datasize = len(train_list)
        self.logger.info('data size: %d' % datasize)
        n_chunk = datasize // chunk_size + 1

        chunk_list = []
        for index in range(n_chunk):
            start_index = index * chunk_size
            end_index = start_index + chunk_size
            chunk = train_list[start_index:end_index]
            chunk_list.append((index, chunk))

        for chunk in chunk_list:
            index = chunk[0]
            chunk = chunk[1]
            st = time.time()
            temp_list = []
            for i, (key, cate_vec, title) in enumerate(chunk):

                title_vec = self.parser.text2vec(title)
                temp_list.append((key, cate_vec, title_vec))

            chunk_path = (os.path.join(self.temp_dir_path, 'products_tv_{}.pkl'.format(index)))
            f_out = open(chunk_path, 'wb')
            p = cPickle.Pickler(f_out)
            p.dump(temp_list)
            p.clear_memo()
            f_out.close()

            self.logger.info("%s done: %d" % (chunk_path, time.time() - st))

        if not os.path.isdir(self.train_dir_path):
            os.makedirs(self.train_dir_path)

        all_train = train_ratio >= 1.0
        all_dev = train_ratio == 0.0

        train_indices, train_size = self.get_train_indices(datasize, train_ratio)

        dev_size = datasize - train_size
        if all_dev:
            train_size = 1
            dev_size = datasize
        if all_train:
            dev_size = 1
            train_size = datasize

        data_fout = h5py.File(os.path.join(self.train_dir_path, 'data.h5py'), 'w')
        train = data_fout.create_group('train')
        dev = data_fout.create_group('dev')
        self.create_dataset(train, train_size, n_target)
        self.create_dataset(dev, dev_size, n_target)

        sample_idx = 0
        dataset = {'train': train, 'dev': dev}
        num_samples = {'train': 0, 'dev': 0}
        chunk = {'train': self.init_chunk(chunk_size, n_target),
                 'dev': self.init_chunk(chunk_size, n_target)}

        # make h5file
        for input_chunk_idx in range(n_chunk):
            path = os.path.join(self.temp_dir_path, 'products_tv_{}.pkl'.format(input_chunk_idx))
            print('processing %s ...' % path)
            data = cPickle.loads(open(path, 'rb').read())
            for data_idx, (img_idx, cate_vec, title_vec) in enumerate(data):

                is_train = train_indices[sample_idx + data_idx]
                if all_dev:
                    is_train = False
                if all_train:
                    is_train = True

                c = chunk['train'] if is_train else chunk['dev']
                idx = c['num']
                c['docvec'][idx] = title_vec
                c['cate'][idx] = cate_vec
                c['asin'].append(np.string_(img_idx))
                c['num'] += 1

                for t in ['train', 'dev']:
                    if chunk[t]['num'] >= chunk_size:
                        self.copy_chunk(dataset[t], chunk[t], num_samples[t])
                        num_samples[t] += chunk[t]['num']
                        chunk[t] = self.init_chunk(chunk_size, n_target)

            sample_idx += len(data)

        for t in ['train', 'dev']:
            if chunk[t]['num'] > 0:
                self.copy_chunk(dataset[t], chunk[t], num_samples[t])
                num_samples[t] += chunk[t]['num']

        for div in ['train', 'dev']:
            ds = dataset[div]
            size = num_samples[div]
            ds['cate'].resize((size, n_target))
            ds['docvec'].resize((size, self.doc_vec_size))

        data_fout.close()


def main(servertype, dataset):
    if dataset == 'products':
        config_path = './configs/config-{}.yaml'.format(servertype)
        config = ges_Aonfig(config_path)
        data = eCommerceData(config)
        data.save_caption_vectors_products(train_ratio=0.8)
    else:
        print('Preprocessor for this dataset is not available.')


if __name__ == '__main__':

    fire.Fire({"make_db": main})
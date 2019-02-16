import os
import fire
import time
import sys
import skipthoughts
import traceback
from sklearn.externals import joblib
import random
import h5py
import numpy as np
from misc import get_logger, ges_Aonfig
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


def save_caption_vectors_flowers(data_dir, dt_range=(1, 103)):
    import time

    img_dir = os.path.join(data_dir, 'flowers/jpg')
    all_caps_dir = os.path.join(data_dir, 'flowers/all_captions.txt')
    target_file_path = os.path.join(data_dir, "flowers/allclasses.txt")
    caption_dir = os.path.join(data_dir, 'flowers/text_c10')
    image_files = [f for f in os.listdir(img_dir) if 'images' in f]
    print(image_files[300 :400])
    image_captions = {}
    image_classes = {}
    class_dirs = []
    class_names = []
    img_ids = []

    target, one_hot_targets, n_target = get_one_hot_targets(target_file_path)

    for i in range(dt_range[0], dt_range[1]) :
        class_dir_name = 'class_%.5d' % (i)
        class_dir = os.path.join(caption_dir, class_dir_name)
        class_names.append(class_dir_name)
        class_dirs.append(class_dir)
        onlyimgfiles = [f[0 :11] + ".jpg" for f in os.listdir(class_dir)
                                    if 'txt' in f]
        for img_file in onlyimgfiles:
            image_classes[img_file] = None

        for img_file in onlyimgfiles:
            image_captions[img_file] = []

    for class_dir, class_name in zip(class_dirs, class_names) :
        caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
        for i, cap_file in enumerate(caption_files) :
            if i%50 == 0:
                print(str(i) + ' captions extracted from' + str(class_dir))
            with open(os.path.join(class_dir, cap_file)) as f :
                str_captions = f.read()
                captions = str_captions.split('\n')
            img_file = cap_file[0 :11] + ".jpg"

            # 5 captions per image
            image_captions[img_file] += [cap for cap in captions if len(cap) > 0][0 :5]
            image_classes[img_file] = one_hot_encode_str_lbl(class_name,
                                                             target,
                                                             one_hot_targets)

    model = skipthoughts.load_model()
    encoded_captions = {}
    for i, img in enumerate(image_captions) :
        st = time.time()
        encoded_captions[img] = skipthoughts.encode(model, image_captions[img])
        if i%20 == 0:
            print(i, len(image_captions), img)
            print("Seconds", time.time() - st)
        if i > 100:
            break

    img_ids = list(image_captions.keys())

    random.shuffle(img_ids)
    n_train_instances = int(len(img_ids) * 0.9)
    tr_image_ids = img_ids[0 :n_train_instances]
    val_image_ids = img_ids[n_train_instances : -1]

    joblib.dump(image_captions, os.path.join(data_dir, 'flowers', 'flowers_caps.pkl'))

    joblib.dump(tr_image_ids, os.path.join(data_dir, 'flowers', 'train_ids.pkl'))
    joblib.dump(val_image_ids, os.path.join(data_dir, 'flowers', 'val_ids.pkl'))

    ec_pkl_path = (os.path.join(data_dir, 'flowers', 'flowers_tv.pkl'))
    joblib.dump(encoded_captions, ec_pkl_path)

    fc_pkl_path = (os.path.join(data_dir, 'flowers', 'flowers_tc.pkl'))
    joblib.dump(image_classes, fc_pkl_path)


def preprocessing(chunk, data_dir):
    index = chunk[0]
    key_caps = chunk[1]
    try:
        model = skipthoughts.load_model()
        titles = [cap[0] for key, cap in key_caps]
        keys = [key for key, cap in key_caps]
        st = time.time()
        encoded_caption_array = skipthoughts.encode(model, titles)
        print("Seconds", time.time() - st)

        encoded_captions = {}
        for i, img in enumerate(keys):
            encoded_captions[img] = encoded_caption_array[i, :].reshape(1, -1)
            # if i > 100:
            #     break

        ec_pkl_path = (os.path.join(data_dir, 'products/tmp', 'products_tv_{}.pkl'.format(index)))
        joblib.dump(encoded_captions, ec_pkl_path)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


class eCommerceData:
    def __init__(self, config):
        self.out_dir = config['DATAPREP']['OUT_DIR']
        self.chunk_size = config['DATAPREP']['CHUNK_SIZE']
        self.num_workers = config['DATAPREP']['NUM_WORKERS']
        self.skipvec_size = config['DATAPREP']['SKIPVEC_SIZE']

    def get_train_indices(self, size, train_ratio):
        train_indices = np.random.rand(size) < train_ratio
        train_size = int(np.count_nonzero(train_indices))
        return train_indices, train_size

    def create_dataset(self, g, size, num_classes):
        g.create_dataset('skipvec', (size, self.skipvec_size), chunks=True, dtype=np.float32)
        g.create_dataset('cate', (size, num_classes), chunks=True, dtype=np.int32)
        g.create_dataset('asin', (size,), chunks=True, dtype='S14')

    def init_chunk(self, chunk_size, num_classes):
        chunk = {}
        chunk['skipvec'] = np.zeros(shape=(chunk_size, self.skipvec_size), dtype=np.float32)
        chunk['cate'] = np.zeros(shape=(chunk_size, num_classes), dtype=np.int32)
        chunk['asin'] = []
        chunk['num'] = 0
        return chunk

    def copy_chunk(self, dataset, chunk, offset):
        num = chunk['num']
        dataset['skipvec'][offset:offset + num] = chunk['skipvec'][:num]
        dataset['cate'][offset:offset + num] = chunk['cate'][:num]
        dataset['asin'][offset:offset + num] = chunk['asin'][:num]

    def save_caption_vectors_products(self, data_dir, train_ratio=0.8):

        caption_path = os.path.join(data_dir, 'products/products.tsv')
        target_file_path = os.path.join(data_dir, "products/categories.txt")
        image_captions = {}
        image_classes = {}

        target, one_hot_targets, n_target = get_one_hot_targets(target_file_path)

        with open(caption_path) as cap_f:
            for i, line in enumerate(cap_f):
                row = line.strip().split('\t')
                # try:
                asin = row[0]
                categories = row[1]
                title = row[2]
                # except:
                #     print(row)

                imgid = asin+'.jpg'
                image_captions[imgid] = [title]
                image_classes[imgid] = np.sum(np.asarray([one_hot_encode_str_lbl(class_name, target, one_hot_targets)
                                               for class_name in categories.split(',')]), axis=0).astype(bool).astype(int)

                if i % 100000 == 0:
                    print(i, image_captions[imgid])
                    print(np.sum(image_classes[imgid]), categories)

                # if i > 20000:
                #     break

        chunk_size = self.chunk_size
        datasize = len(image_captions)
        print(datasize)
        n_chunk = datasize // chunk_size + 1
        image_captions_list = list(image_captions.items())
        chunk_list = []
        for index in range(n_chunk):
            start_index = index * chunk_size
            end_index = start_index + chunk_size
            chunk = image_captions_list[start_index:end_index]
            chunk_list.append((index, chunk))

        model = skipthoughts.load_model()
        for chunk in chunk_list:
            index = chunk[0]
            key_caps = chunk[1]
            titles = [cap[0] for key, cap in key_caps]
            keys = [key for key, cap in key_caps]
            st = time.time()
            encoded_caption_array = skipthoughts.encode(model, titles)
            print("Seconds", time.time() - st)

            encoded_captions = {}
            for i, img in enumerate(keys):
                encoded_captions[img] = encoded_caption_array[i, :].reshape(1, -1)
                # if i > 100:
                #     break

            # ec_pkl_path = (os.path.join(data_dir, 'products/tmp', 'products_tv_{}.pkl'.format(index)))
            # f_out = open(ec_pkl_path, 'wb')
            # p = cPickle.Pickler(f_out)
            # p.dump(encoded_captions)
            # p.clear_memo()
            # f_out.close()


            del encoded_captions
            del encoded_caption_array


        # pool = Pool(self.num_workers)
        # try:
        #     pool.map_async(partial(preprocessing, data_dir=data_dir), chunk_list).get(999999999)
        #     pool.close()
        #     pool.join()
        # except KeyboardInterrupt:
        #     pool.terminate()
        #     pool.join()
        #     raise

        joblib.dump(image_captions, os.path.join(data_dir, 'products', 'products_caps.pkl'))

        fc_pkl_path = (os.path.join(data_dir, 'products', 'products_tc.pkl'))
        joblib.dump(image_classes, fc_pkl_path)

        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

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

        data_fout = h5py.File(os.path.join(self.out_dir, 'data.h5py'), 'w')
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
            path = os.path.join(data_dir, 'products/tmp', 'products_tv_{}.pkl'.format(input_chunk_idx))
            print('processing %s ...' % path)
            data = cPickle.loads(open(path, 'rb').read())
            for data_idx, (img_idx, enc_cap) in enumerate(data.items()):
                cate = image_classes[img_idx]

                is_train = train_indices[sample_idx + data_idx]
                if all_dev:
                    is_train = False
                if all_train:
                    is_train = True

                c = chunk['train'] if is_train else chunk['dev']
                idx = c['num']
                c['skipvec'][idx] = enc_cap
                c['cate'][idx] = cate
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
            ds['skipvec'].resize((size, self.skipvec_size))

        data_fout.close()


def main(servertype, datadir, dataset):
    dataset_dir = os.path.join(datadir, "datasets")
    if dataset == 'flowers':
        save_caption_vectors_flowers(dataset_dir)
    if dataset == 'products':
        config_path = './configs/config-{}.yaml'.format(servertype)
        config = ges_Aonfig(config_path)
        data = eCommerceData(config)
        data.save_caption_vectors_products(dataset_dir)
    else:
        print('Preprocessor for this dataset is not available.')


if __name__ == '__main__':

    fire.Fire({"make_db": main})
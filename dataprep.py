import os
import fire
import time
import glob
import argparse
import skipthoughts
import traceback
import pickle
import random

import numpy as np


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

    pickle.dump(image_captions,
                open(os.path.join(data_dir, 'flowers', 'flowers_caps.pkl'), "wb"))

    pickle.dump(tr_image_ids,
                open(os.path.join(data_dir, 'flowers', 'train_ids.pkl'), "wb"))
    pickle.dump(val_image_ids,
                open(os.path.join(data_dir, 'flowers', 'val_ids.pkl'), "wb"))

    ec_pkl_path = (os.path.join(data_dir, 'flowers', 'flowers_tv.pkl'))
    pickle.dump(encoded_captions, open(ec_pkl_path, "wb"))

    fc_pkl_path = (os.path.join(data_dir, 'flowers', 'flowers_tc.pkl'))
    pickle.dump(image_classes, open(fc_pkl_path, "wb"))


def save_caption_vectors_products(data_dir):
    caption_path = os.path.join(data_dir, 'products/products.tsv')
    target_file_path = os.path.join(data_dir, "products/categories.txt")
    image_captions = {}
    image_classes = {}

    target, one_hot_targets, n_target = get_one_hot_targets(target_file_path)

    with open(caption_path) as cap_f:
        for i, line in enumerate(cap_f):
            row = line.strip().split('\t')
            try:
                asin = row[0]
                categories = row[1]
                title = row[2]
            except:
                print(row)

            imgid = asin+'.jpg'
            image_captions[imgid] = [title]
            image_classes[imgid] = np.sum(np.asarray([one_hot_encode_str_lbl(class_name, target, one_hot_targets)
                                           for class_name in categories.split(',')]), axis=0)

            if i % 100000 == 0:
                print(i, image_captions[imgid])
                print(np.sum(image_classes[imgid]), categories)

            # if i > 100:
            #     break

    chunk_size = 500000
    n_chunk = len(image_captions) // 500000 + 1
    model = skipthoughts.load_model()
    for i in range(n_chunk):
        start_index = i*chunk_size
        end_index = start_index + chunk_size
        titles = [j for sub in image_captions.values()[start_index:end_index] for j in sub]
        st = time.time()
        encoded_caption_array = skipthoughts.encode(model, titles)
        print("Seconds", time.time() - st)

        encoded_captions = {}
        for i, img in enumerate(image_captions.keys()[start_index:end_index]):
            encoded_captions[img] = encoded_caption_array[i, :].reshape(1, -1)
            # if i > 100:
            #     break

        ec_pkl_path = (os.path.join(data_dir, 'products/tmp', 'products_tv_{}.pkl'.format(i)))
        pickle.dump(encoded_captions, open(ec_pkl_path, "wb"))

    # img_ids = list(image_captions.keys())
    #
    # random.shuffle(img_ids)
    # n_train_instances = int(len(img_ids) * 0.9)
    # tr_image_ids = img_ids[0:n_train_instances]
    # val_image_ids = img_ids[n_train_instances: -1]
    #
    # pickle.dump(image_captions,
    #             open(os.path.join(data_dir, 'products', 'products_caps.pkl'), "wb"))
    #
    # pickle.dump(tr_image_ids,
    #             open(os.path.join(data_dir, 'products', 'train_ids.pkl'), "wb"))
    # pickle.dump(val_image_ids,
    #             open(os.path.join(data_dir, 'products', 'val_ids.pkl'), "wb"))
    #
    # fc_pkl_path = (os.path.join(data_dir, 'products', 'products_tc.pkl'))
    # pickle.dump(image_classes, open(fc_pkl_path, "wb"))


def main(datadir, dataset):
    dataset_dir = os.path.join(datadir, "datasets")
    if dataset == 'flowers':
        save_caption_vectors_flowers(dataset_dir)
    if dataset == 'products':
        save_caption_vectors_products(dataset_dir)
    else:
        print('Preprocessor for this dataset is not available.')


if __name__ == '__main__':
    fire.Fire({"main": main})
import gzip
import os
import re
import tqdm
import fire
import time
import random
import numpy as np
from collections import Counter
import sentencepiece as spm
from misc import get_logger, ges_Aonfig
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def shuffle(ls):
    random.shuffle(ls)
    return ls


class EcommerceDataParser:
    def __init__(self, config, use=False):
        self.logger = get_logger()
        self.meta_path = config['META_PATH']
        self.titles_path = config['TITLES_PATH']
        self.spm_dir_path = config['SPM_DIR_PATH']
        self.spm_wp_path = config['SPM_WP_PATH']
        self.category_path = config['CATEGORY_PATH']
        self.parse_data_path = config['PARSE_DATA_PATH']
        self.doc2vec_dir_path = config['DOC2VEC_DIR_PATH']
        self.use_cols = config['USE_COLS']
        self.use_cate = config['USE_CATE']
        self.n_sample = config['N_SAMPLE']
        self.vocab_size = config['VOCAB_SIZE']
        self.n_shuffle = config['N_SHUFFLE']
        self.cate_depth = config['CATE_DEPTH']
        self.n_log_print = config['N_LOG_PRINT']

        self.doc_vec_size = config['DOC_VEC_SIZE']
        self.doc2vec_epochs = config['DOC2CEC_EPOCHS']
        self.n_workers = config['N_WORKERS']
        self.window_size = config['WINDOW_SIZE']

        self.re_sc = re.compile('[\!@#$%\^&\*\(\)=\[\]\{\}\.,/\?~\+\"|\_\-:;]')
        self.stopwords =['&amp;', '&quot;']
        if use:
            self.load_spm()
            self.load_doc2vec_model()

    def load_spm(self):
        self.logger.info('USE MODE LOAD SPM')
        st = time.time()
        self.sp = spm.SentencePieceProcessor()
        spm_model_path = os.path.join(self.spm_dir_path, 'spm.model')
        self.sp.Load(spm_model_path)
        self.i2wp = [line.split('\t')[0] for line in open(self.spm_wp_path)]
        self.wp2i = dict([(v, i) for i, v in enumerate(self.i2wp)])
        self.logger.info('USE MODE LOAD SPM DONE: %d sec' % (time.time() - st))

    def preprocess(self):
        t = time.time()
        self.logger.info('PARSE START')
        self.parse_data()
        self.logger.info('PARSE DONE: %d sec' % (time.time() - t))
        t = time.time()
        self.logger.info('TRAIN SPM')
        self.train_spm()
        self.logger.info('TRAIN SPM DONE: %d sec' % (time.time() - t))
        self.logger.info('BUILD SPM WP VOCAB')
        self.build_x_vocab(self.titles_path, self.spm_dir_path, self.spm_wp_path)
        self.logger.info('BUILD SPM WP VOCAB DONE: %d sec' % (time.time() - t))
        self.load_spm()
        self.logger.info(self.text2wp('adult ballet tutu cheetah pink'))
        self.logger.info('TRAIN DOC2VEC')
        self.train_doc2vec()
        self.logger.info('TRAIN DOC2VEC DONE: %d sec' % (time.time() - t))
        q = self.text2wp('samsung notebook ssd 256gb ram 8gb')
        self.query_doc2vec_topn(q)

    def remove_stopwords(self, text):
        for stopword in self.stopwords:
            text = text.replace(stopword, '')
        return text

    def text_cleaning(self, text):
        text = self.remove_stopwords(text)
        text = self.re_sc.sub(' ', text).strip()
        return ' '.join(text.split()).lower()

    def write_titles(self, titles):
        titles_dir = os.path.dirname(self.titles_path)
        os.makedirs(titles_dir, exist_ok=True)

        f_titles = open(self.titles_path, 'w')

        for title in tqdm.tqdm(titles, mininterval=1):
            f_titles.write(title + '\n')

    def parse_data(self):
        data_list = []
        category_list = []
        titles = ['text']
        g = gzip.open(self.meta_path, 'r')
        for i, l in enumerate(g):
            product = eval(l)
            n_read = int(self.n_log_print) * 10
            if i % n_read == 0:
                self.logger.info("Read %d lines..." % i)

            skip_flag = False

            for col in self.use_cols:
                if col not in product.keys():
                    skip_flag = True

            if skip_flag:
                continue

            asin = product['asin']
            url = product['imUrl']
            brand = product['brand'] if 'brand' in product.keys() else ''
            catenames = ' '.join(list(map(lambda x: ' '.join(shuffle(x[1:])), product['categories'])))

            raw_categories = product['categories'][0] if len(product['categories']) > 0 else None

            if raw_categories is None:
                continue

            if len(self.use_cate) > 0 and raw_categories[0] not in self.use_cate:
                continue

            raw_categories = list(map(lambda x: x.replace('>', '').replace(' ', '').strip(),
                                      raw_categories[:self.cate_depth]))
            category = '>'.join(raw_categories)

            # hardcoding erase cate
            chose_flag = False

            cates = category.split('>')
            cates = cates[1] if len(cates) > 2 else None
            if cates is not None and 'Guitars' in cates:
                chose_flag = True

            if 'BeginnerKits' in category:
                chose_flag = False

            # select_cates = [
            #       "Clothing,Shoes&Jewelry>adidas"
            #     , "Home&Kitchen>Furniture>LivingRoomFurniture>Tables"
            #     , "Clothing,Shoes&Jewelry>Women>Clothing>Coats&Jackets"
            #     , "Beauty>Makeup>Lips>Lipstick"
            #     , "Clothing,Shoes&Jewelry>Women>Shoes>Boots"
            #     , "Clothing,Shoes&Jewelry>Girls>Clothing>Dresses"
            #     , "Clothing,Shoes&Jewelry>Women>Accessories>Hats&Caps"
            #     , "Clothing,Shoes&Jewelry>Women>Clothing>Skirts"
            #     , "Clothing,Shoes&Jewelry>Women>Handbags&Wallets>ShoulderBags"
            #     , "Automotive>Motorcycle&Powersports>ProtectiveGear>Helmets"
            #     , "Clothing,Shoes&Jewelry>N>Nike"
            #     , "Tools&HomeImprovement>Lighting&CeilingFans>Lamps&Shades>TableLamps"
            # ]
            # if category in select_cates:
            #     chose_flag = True

            if not chose_flag:
                continue

            title = self.text_cleaning(' '.join(shuffle([catenames, brand, product['title']])))
            if len(title) == 0:
                continue

            shuffle_titles = set()
            shuffle_titles.add(title)
            for _ in range(self.n_shuffle):
                shuffle_titles.add(self.text_cleaning(' '.join(shuffle([catenames, brand, product['title']]))))

            shuffle_titles = list(shuffle_titles)
            titles += shuffle_titles
            for title in shuffle_titles:
                data_list.append((asin, category, title, url))

            if category not in category_list:
                category_list.append(category)

            if len(data_list) % self.n_log_print == 0:
                self.logger.info("%s\t%s\t%s %s -> %s [%s]" % (i, asin, product['title'], product['categories'], title, category))

            if i > self.n_sample:
                break

        self.write_titles(titles)

        with open(self.parse_data_path, 'w') as data_file:
            for data in data_list:
                output = "{}\t{}\t{}\t{}\n".format(data[0], data[1], data[2], data[3])
                data_file.write(output)

        with open(self.category_path, 'w') as data_file:
            for category in category_list:
                output = "{}\n".format(category)
                data_file.write(output)

    def train_spm(self, input_sentence_size=10000000):
        spm_path = os.path.join(self.spm_dir_path, 'spm')
        txt_path = self.titles_path
        vocab_size = self.vocab_size
        spm_dir = os.path.dirname(spm_path)
        os.makedirs(spm_dir, exist_ok=True)
        spm.SentencePieceTrainer.Train(
            f' --input={txt_path} --model_type=bpe'
            f' --model_prefix={spm_path} --vocab_size={vocab_size}'
            f' --input_sentence_size={input_sentence_size}')

    def write_vocab(self, vocab, vocab_fn):
        with open(vocab_fn, 'w') as fp:
            for v, c in vocab:
                fp.write(f'{v}\t{c}\n')

    def build_x_vocab(self, txt_path, spm_dir_path, wp_vocab_path):
        spm_model_path = os.path.join(spm_dir_path, 'spm.model')
        sp = spm.SentencePieceProcessor()
        sp.Load(spm_model_path)

        wp_counter = Counter()
        title_lines = open(txt_path).readlines()

        max_wps_len = 0
        max_words_len = 0
        for line in tqdm.tqdm(title_lines, mininterval=1):
            line = line.strip()
            words = line.split()
            max_words_len = max(max_words_len, len(words))

            wps = []
            for w in words:
                wp = sp.EncodeAsPieces(w)
                max_wps_len = max(len(wp), max_wps_len)
                wps += wp

            for wp in wps:
                wp_counter[wp] += 1

        wp_vocab = [('PAD', max_wps_len)] + wp_counter.most_common()
        self.write_vocab(wp_vocab, wp_vocab_path)

    def text2wp(self, text):
        words = text.split()
        wp_sent = []
        for i, word in enumerate(words):
            wps = self.sp.EncodeAsPieces(word)
            wp_indices = [self.wp2i[wp] for wp in wps if wp in self.wp2i]
            wp_sent += wp_indices

        return wp_sent

    def get_doc_list(self):
        self.document_list = []
        with open(self.parse_data_path, 'r') as data_file:
            st = time.time()
            for index, data in enumerate(data_file):
                data = data.split('\t')
                key = data[0] + '.jpg'
                cate = data[1]
                title = data[2]
                wp_i = self.text2wp(title)
                i_wp = [self.i2wp[i] for i in wp_i]
                if index % self.n_log_print == 0:
                    self.logger.info("%s %s %s %d sec" % (title, i_wp, wp_i, time.time() - st))
                    st = time.time()
                wp_i_str = list(map(lambda x: str(x), wp_i))
                self.document_list.append((key, wp_i_str))

    def train_doc2vec(self):
        import logging
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO)

        doc2vec_model_path = os.path.join(self.doc2vec_dir_path, 'doc2vec.model')

        self.get_doc_list()
        documents = [TaggedDocument(doc, [key]) for key, doc in self.document_list]
        self.model = Doc2Vec(documents, vector_size=self.doc_vec_size,
                        window=self.window_size,
                        min_count=1,
                        workers=self.n_workers,
                        max_vocab_size=None,
                        epochs=self.doc2vec_epochs)

        self.model.save(doc2vec_model_path)

    def load_doc2vec_model(self):
        st = time.time()
        self.logger.info('USE MODE LOAD DOC2VEC')
        doc2vec_model_path = os.path.join(self.doc2vec_dir_path, 'doc2vec.model')
        self.get_doc_list()
        self.model = Doc2Vec.load(doc2vec_model_path)
        self.logger.info('USE MODE LOAD DOC2VEC DONE: %d sec' % (time.time() - st))

    def search_doc(self, q_key):
        for key, doc  in self.document_list:
            if key == q_key:
                return [self.i2wp[int(i)] for i in doc]
        return ['NO_SEARCH_RESULT']

    def query_doc2vec_topn(self, q):
        if type(q) == str:
            q = q.split()

        q = np.asarray(q).astype(str)

        vector = self.model.infer_vector(q)
        sims = self.model.docvecs.most_similar([vector])

        print(''.join([self.i2wp[int(i)] for i in q]))
        print()
        print('[TOP-N SIM]')
        for i, (key, score) in enumerate(sims):
            print('%s\t%s\t%.2f' % (i, ''.join(self.search_doc(key)), score))
        print()

    def text2vec(self, text):
        wps = self.text2wp(text)
        wps_str = list(map(lambda x: str(x), wps))
        vector = self.model.infer_vector(wps_str)
        return vector


def main(config_path):
    config = ges_Aonfig(config_path)['PARSEMETA']
    parser = EcommerceDataParser(config)
    parser.preprocess()
    print(parser.text2wp('adult ballet tutu cheetah pink'))


if __name__ == '__main__':
    fire.Fire({'parse': main})

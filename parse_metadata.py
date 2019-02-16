import gzip
import os
import tqdm

def write_titles(titles, titles_path):
    titles_dir = os.path.dirname(titles_path)
    os.makedirs(titles_dir, exist_ok=True)

    f_titles = open(titles_path, 'w')

    for title in tqdm.tqdm(titles, mininterval=1):
        f_titles.write(title + '\n')

def parse_data(path):
    data_list = []
    category_list = []
    titles = ['text']
    g = gzip.open(path, 'r')
    for i, l in enumerate(g):
        product = eval(l)

        skip_flag = False
        for col in ['asin', 'imUrl', 'title', 'categories']:
            if col not in product.keys():
                skip_flag = True

        if skip_flag:
            continue

        asin = product['asin']
        raw_categories = sum(list(map(lambda x:[x[0]] if len(x) > 0 else ['-1'], product['categories'])), [])
        raw_categories = list(map(lambda x: ' '.join(x.replace('\n', ' ').replace('\t', ' ').replace(',', ' ').split())
                                  if len(x.replace(' ', '')) > 0 else '-1', raw_categories))
        categories = ",".join(raw_categories)
        title = product['title'].replace('\n', ' ').replace('\t', ' ').replace(',', ' ')\
            .replace('&amp;', '').replace('&quot;', '')
        if len(title) == 0:
            continue

        titles.append(title)
        data_list.append((asin, categories, title))

        for category in raw_categories:
            if category not in category_list:
                category_list.append(category)

        if i % 10000 == 0:
            print(i, product['asin'], product['title'], product['categories'])

    write_titles(titles, 'data/datasets/products.csv')

    with open('./data/datasets/products/products.tsv', 'w') as data_file:
        for data in data_list:
            output = "{}\t{}\t{}\n".format(data[0], data[1], data[2])
            data_file.write(output)

    with open('./data/datasets/products/categories.txt', 'w') as data_file:
        for category in category_list:
            output = "{}\n".format(category)
            data_file.write(output)


if __name__ == '__main__':
    parse_data('./data/datasets/products/metadata.json.gz')

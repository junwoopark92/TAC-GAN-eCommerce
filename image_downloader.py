import sys, os, multiprocessing, urllib3
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import gzip
from multiprocessing.pool import ThreadPool

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def parse_data(path='data/datasets/products/products.tsv'):
    key_url_list = []
    with open(path) as f:
        for i, l in enumerate(f):
            row = l.split('\t')
            asin = row[0]
            url = row[3].strip() if len(row) > 3 else None
            if i % 10000 == 0:
                print('read lines: {}'.format(i))

            key_url_list.append((asin, url))
            if i % 1000 == 0:
                print(i,row)

    return key_url_list


def download_image(key_url):
    outdir = sys.argv[2]
    key, url = key_url
    filename = os.path.join(outdir, '%s.jpg' % key)

    if os.path.exists(filename):
        print('Image %s already exists. Skipping download.' % filename)
        return

    try:
        # print('Trying to get %s.' % url)
        http = urllib3.PoolManager()
        response = http.request('GET', url)
        image_data = response.data
    except Exception as e:
        print(e)
        print('Warning: Could not download image %s from %s' % (key, url))
        return

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image %s %s' % (key, url))
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image %s to RGB' % key)
        return

    try:
        pil_image_rgb.save(filename, format='JPEG')
    except:
        print('Warning: Failed to save image %s' % filename)
        return


def run():
    if len(sys.argv) != 3:
        print('Syntax: %s <train|validation|test.json> <output_dir/>' % sys.argv[0])
        sys.exit(0)

    data_file, out_dir = sys.argv[1:]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = parse_data(data_file)

    pool = ThreadPool(processes=20)

    with tqdm(total=len(key_url_list)) as t:
        for _ in pool.imap_unordered(download_image, key_url_list):
            t.update(1)


if __name__ == '__main__':
    run()
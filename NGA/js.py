import os
import json
import argparse

def merge(filename, sort):
    files = os.listdir(filename)
    if sort == 1:
        files.sort(key=lambda x:int(x[:-5]))
    fin = []
    for file in files:
        f = open(filename + '/' + file, encoding='utf-8')
        data = json.load(f, encoding='utf-8')
        fin += data

    with open(filename + '.json', 'w', encoding='utf-8') as f:
        json.dump(fin, f, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='', help='file path')
    parser.add_argument('--sort', type=int, default=1, help='sort mode')
    args = parser.parse_args()
    if args.path != '':
        merge(args.path, args.sort)

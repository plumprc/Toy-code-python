import os
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='./', help='file name')
    args = parser.parse_args()
    files = os.listdir(args.file)
    fin = []
    for i in range(len(files)):
        f = open(args.file + '/' + str(i+1) + '.json', encoding='utf-8')
        data = json.load(f, encoding='utf-8')
        fin += data

    with open(args.file + '.json', 'w', encoding='utf-8') as f:
        json.dump(fin, f, ensure_ascii=False)

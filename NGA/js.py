import re
import os
import json
import argparse

def merge(filename):
    files = os.listdir(filename)
    # files.sort(key=lambda x:int(x[:-5]))
    fin = []
    for file in files:
        f = open(filename + '/' + file, encoding='utf-8')
        data = json.load(f, encoding='utf-8')
        fin += data

    with open(filename + '.json', 'w', encoding='utf-8') as f:
        json.dump(fin, f, ensure_ascii=False)

def clean_quote(filename: str):
    f = open(f'{filename}.json', encoding='utf-8')
    data = json.load(f)
    hash = set()
    for reply in data:
        for idx in range(len(reply['reply'])):
            s = reply['reply'][idx]
            if s not in hash:
                hash.add(s)
                if s.find('quote') != -1:
                    reply['reply'][idx] = ''
                    continue
            else:
                reply['reply'][idx] = ''
    del(hash)
    for reply in data:
        rep = filter(lambda x : x != '', reply['reply'])
        reply['reply'] = list(rep)

    with open(filename + '.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def clean_reply(filename: str):
    f = open(f'{filename}.json', encoding='utf-8')
    data = json.load(f)
    for reply in data:
        for idx in range(len(reply['reply'])):
            s = reply['reply'][idx]
            r = re.findall(r'\[b\][\s\S]+?\[\/b\]', s)
            if len(r) != 0:
                for con in r:
                    uid = re.findall(r'\[uid=-?\d+\]', con)
                    if len(uid) != 0:
                        s = s.replace(con, 'rep' + uid[0][1:-1] + ' ')
            reply['reply'][idx] = s
    
    with open(filename + '.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='', help='file name')
    parser.add_argument('--path', type=str, default='', help='file path')
    args = parser.parse_args()
    if args.file != '':
        filename = 'statistics/' + args.file
        clean_quote(filename)
        clean_reply(filename)
        print('clean!')
    if args.path != '':
        merge(args.path)

import json
import re
import sys

def generate_context(filename: str):
    f = open(f'{filename}.json', encoding='utf-8')
    data = json.load(f)
    for reply in data:
        for s in reply['reply']:
            r = re.findall(r'\[s:\S*?\]', s)
            if len(r) != 0:
                for con in r:
                    s = s.replace(con, '')
            r = re.findall(r'repuid=[0-9]+', s)
            if len(r) != 0:
                for con in r:
                    s = s.replace(con, '')
            r = re.findall(r'\[collapse=\S+?\]', s)
            if len(r) != 0:
                for con in r:
                    s = s.replace(con, '')
            r = re.findall(r'\[img\S+?\/img\]', s)
            if len(r) != 0:
                for con in r:
                    s = s.replace(con, '')
            r = re.findall(r'\[url\S+?\/url\]', s)
            if len(r) != 0:
                for con in r:
                    s = s.replace(con, '')
            r = re.findall(r'\[b\][\s\S]+?\[\/b', s)
            if len(r) != 0:
                for con in r:
                    s = s.replace(con, '')

            s = s.replace('[del]', '')
            s = s.replace('[/del]', '')
            s = s.replace('[color]', '')
            s = s.replace('[/color]', '')
            s = s.replace('[collapse]', '')
            s = s.replace('[/collapse]', '')

            with open(filename + '.txt', 'a+', encoding='utf-8') as f:
                f.write(s + '\n')

def generate_context_by_uid(filename: str, uid):
    f = open(f'{filename}.json', encoding='utf-8')
    data = json.load(f)
    for reply in data:
        if uid == str(reply['uid']):
            for s in reply['reply']:
                r = re.findall(r'\[s:\S*?\]', s)
                if len(r) != 0:
                    for con in r:
                        s = s.replace(con, '')
                r = re.findall(r'repuid=[0-9]+', s)
                if len(r) != 0:
                    for con in r:
                        s = s.replace(con, '')
                r = re.findall(r'\[collapse=\S+?\]', s)
                if len(r) != 0:
                    for con in r:
                        s = s.replace(con, '')
                r = re.findall(r'\[img\S+?\/img\]', s)
                if len(r) != 0:
                    for con in r:
                        s = s.replace(con, '')
                r = re.findall(r'\[url\S+?\/url\]', s)
                if len(r) != 0:
                    for con in r:
                        s = s.replace(con, '')
                r = re.findall(r'\[b\][\s\S]+?\[\/b', s)
                if len(r) != 0:
                    for con in r:
                        s = s.replace(con, '')

                s = s.replace('[del]', '')
                s = s.replace('[/del]', '')
                s = s.replace('[color]', '')
                s = s.replace('[/color]', '')
                s = s.replace('[collapse]', '')
                s = s.replace('[/collapse]', '')

                with open(str(uid) + '.txt', 'a+', encoding='utf-8') as f:
                    f.write(s + '\n')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("please input file or uid")
    elif len(sys.argv) == 2:
        filename = 'statistics/' + sys.argv[1]
        generate_context(filename)
    else:
        filename = 'statistics/' + sys.argv[1]
        uid = sys.argv[2]
        generate_context_by_uid(filename, uid)

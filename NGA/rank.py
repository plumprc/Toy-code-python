import json
import re
import datetime

def name(uid):
    with open('uid2name.json', 'r', encoding='utf-8') as f:
        name = json.load(f)
    if str(uid) in name:
        return name[str(uid)]
    
    return str(uid)

class Rank():
    def __init__(self, file_name):
        self.file = open(file_name, encoding='utf-8')
        self.data = json.load(self.file, encoding='utf-8')

    def get_rank(self):
        dic = {}
        for item in self.data:
            if item['uid'] not in dic:
                dic[item['uid']] = 1
            else: dic[item['uid']] += 1
        dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)
        return dic

    def get_time_seq(self):
        dic = {str(i).zfill(2):0 for i in range(24)}
        date = set()
        for item in self.data:
            dic[item['date'][0][11:13]] += 1
            date.add(item['date'][0][5:10])
        
        # dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)
        return dic, len(date)

    def get_calen_flow(self):
        dic = {}
        for item in self.data:
            if item['date'][0][5:10] not in dic:
                dic[item['date'][0][5:10]] = 1
            else: dic[item['date'][0][5:10]] += 1
        
        return dic

    def get_date_act(self):
        dic = {}
        for item in self.data:
            if item['date'][0][5:10] not in dic:
                dic[item['date'][0][5:10]] = set()
                dic[item['date'][0][5:10]].add(item['uid'])
            else: dic[item['date'][0][5:10]].add(item['uid'])
        
        for k, v in dic.items():
            dic[k] = len(v)
        return dic

    def get_meme_freq(self):
        dic = {}
        for item in self.data:
            text = item['reply']
            for con in text:
                r = re.findall(r"\[s:\S*?\]", con)
                for meme in r:
                    if meme not in dic:
                        dic[meme] = 1
                    else: dic[meme] += 1
        
        dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)
        return dic

    def get_meme_uid(self, meme):
        dic = {}
        for item in self.data:
            text = item['reply']
            for con in text:
                r = re.findall(meme, con)
                if len(r) != 0:
                    if item['uid'] not in dic:
                        dic[item['uid']] = len(r)
                    else: dic[item['uid']] += len(r)
        
        dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)
        return dic

    def get_uid_meme(self, uid):
        dic = {}
        for item in self.data:
            if uid != item['uid']:
                continue
            text = item['reply']
            for con in text:
                r = re.findall('\[s:\S*?\]', con)
                if len(r) != 0:
                    for meme in r:
                        if meme not in dic:
                            dic[meme] = 1
                        else: dic[meme] += 1

        dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)
        return dic

    def get_match_reply(self, match):
        for item in self.data:
            text = item['reply']
            for con in text:
                r = re.findall(match, con)
                if len(r) != 0:
                    with open(str(match) + '.txt', 'a+', encoding='utf-8') as f:
                        f.write(str(item['uid']) + ' #' + str(item['floor']) + ' P' + str(item['floor'] // 20 + 1) + ' ' + con + '\n')

        return 1

    def get_uid_reply(self, uid):
        for item in self.data:
            if uid == str(item['uid']):
                text = item['reply']
                for con in text:
                    with open(str(uid) + '.txt', 'a+', encoding='utf-8') as f:
                        f.write('#' + str(item['floor']) + ' P' + str(item['floor'] // 20 + 1) + ' ' + con + '\n')
        
        return 1

    def get_live(self):
        d1 = datetime.datetime.strptime(self.data[1]['date'][0], '%Y-%m-%d %H:%M')
        d2 = datetime.datetime.strptime(self.data[-1]['date'][0], '%Y-%m-%d %H:%M')
        print(d1, d2)
        return d2 - d1

    def get_new_user(self):
        dic = {}
        total = set()
        for item in self.data:
            if item['uid'] not in total:
                total.add(item['uid'])
                if item['date'][0][5:10] not in dic:
                    dic[item['date'][0][5:10]] = set()
                    dic[item['date'][0][5:10]].add(item['uid'])
                else: dic[item['date'][0][5:10]].add(item['uid'])
            else: continue
        
        del(total)
        for k, v in dic.items():
            dic[k] = len(v)
        return dic

    def get_locate(self, loc):
        for item in self.data:
            if loc == item['date'][0][5:10]:
                print(item['floor'])
                break

        return 1

    def get_ipt(self):
        for item in self.data:
            text = item['reply']                
            for con in text:
                r = re.findall("\[同传\]", con)
                if len(r) != 0:
                    with open('ipt.txt', 'a+', encoding='utf-8') as f:
                        f.write(name(item['uid']) + ' #' + str(item['floor']) + '\n')
                    for con in text:
                        with open('ipt.txt', 'a+', encoding='utf-8') as f:
                            f.write(con + '\n')

                    with open('ipt.txt', 'a+', encoding='utf-8') as f:
                        f.write('=========\n\n')
                    
                    break
        
        return 1

import json

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
        for item in self.data:
            dic[item['date'][0][11:13]] += 1
        
        # dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)
        return dic

    def get_calen_flow(self):
        dic = {}
        for item in self.data:
            if item['date'][0][5:10] not in dic:
                dic[item['date'][0][5:10]] = 1
            else: dic[item['date'][0][5:10]] += 1
        
        return dic

    def get_meme_freq(self):
        dic = {}
        return dic

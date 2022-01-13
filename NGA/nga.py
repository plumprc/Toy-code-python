import re
import os
import time
import json
import requests
from tqdm import tqdm
from lxml import etree
import argparse

def clean_quote(reply_list):
    if reply_list[0].find('[quote]') == -1:
        return reply_list
    else:
        if reply_list[0].find('[/pid') == -1:
            return reply_list
        r = re.findall('uid=[0-9]+?\]', reply_list[0])
        if len(r) != 0:
            Ruid = 'R' + r[0][3:-1]
        else: 
            anony = re.findall('anony_[\S]+?\[', reply_list[0])
            if len(anony) != 0:
                Ruid = 'R=' + anony[0][:-1]
            else: Ruid = 'R=anony'
        cnt = 1
        for con in reply_list:
            if con.find('/quote') != -1:
                break
            
            cnt += 1
        if cnt == len(reply_list) + 1:
            return [Ruid] + reply_list[1:]
        
        return [Ruid] + reply_list[cnt:]

def clean_reply(reply_list):
    for idx in range(len(reply_list)):
        s = reply_list[idx]
        r = re.findall(r'\[b\][\s\S]+?\[\/b', s)
        if len(r) != 0:
            for con in r:
                uid = re.findall(r'\[uid=-?\d+\]', con)
                if len(uid) != 0:
                    s = s.replace(con, 'R' + uid[0][4:-1] + ' ')
                    continue
                anony = re.findall(r'anony_[\S]+?\[', con)
                if len(anony) != 0:\
                    s = s.replace(con, 'R=' + anony[0][:-1] + ' ')
                
        r = re.findall(r'\[pid[\s\S]+?\[\/pid\]', s)
        if len(r) != 0:
            for con in r:
                s = s.replace(con, '')
    
        reply_list[idx] = s

    return reply_list

def clean_s(reply_list):
    for idx in range(len(reply_list)):
        s = reply_list[idx]
        r = re.findall(r'四哈人基建公约', s)
        if len(r) != 0:
            reply_list = reply_list[:idx]
            break
    return reply_list    

class NGA(object):
    def __init__(self):
        self.headers = {
            'Connection': 'keep-alive',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36',
            'Cookie': 'taihe_bi_sdk_uid=9f6b3350fa4f95f39ae629d7d6d2433d; taihe=a1747dcebbecee5a50f8dfa53cd7005c; UM_distinctid=177b9ce533354b-0a9bab952bd6fb-73e356b-144000-177b9ce5334617; CNZZDATA30043604=cnzz_eid=1103191160-1594714273-https%3A%2F%2Fbbs.nga.cn%2F&ntime=1623134901; ngacn0comUserInfo=Drelf	Drelf	39	39		10	200	4	0	0	61_2; ngacn0comUserInfoCheck=731861771366f61e88f3af515823dbec; ngacn0comInfoCheckTime=1623137434; ngaPassportUid=61710301; ngaPassportUrlencodedUname=Drelf; ngaPassportCid=X95hemmhg11gme0ll3gusboc7ilo2r3p6fbcu0am; lastvisit=1623137722; lastpath=/thread.php?fid=734&ff=-34587507; bbsmisccookies={"uisetting":{0:0,1:1624459883},"pv_count_for_insad":{0:-157,1:1623171704},"insad_views":{0:2,1:1623171704}}; _cnzz_CV30043604=forum|fid-34587507|0'
        }

    def get_reply(self, tid: str, start: int, file: str='merge') -> int:
        try:
            r = requests.get(f'https://bbs.nga.cn/read.php?tid={tid}', headers=self.headers)
        except Exception:
            return 101
        try:
            total = int(re.search(r"',\d+:\d+", r.text).group().split(':')[1])
        except Exception:
            total = 1

        if start == 1:
            main = etree.HTML(r.text).xpath('//*[@id="postcontent0"]/text()')
            file = etree.HTML(r.text).xpath('//*[@id="postsubject0"]/text()')[0]
        
        if not os.path.exists(file):
            os.mkdir(file)
        for i in tqdm(range(start, total+1)):
            s = []
            try:
                r = requests.get(f'https://bbs.nga.cn/read.php?tid={tid}&page={i}', headers=self.headers)
            except Exception:
                return 102
            if r.status_code == 200:
                data = etree.HTML(r.text)
                for post in data.xpath('//table[@class="forumbox postbox"]'):
                    uid, floor = post.xpath('.//a[@class="author b"]/@*')[:-1]
                    floor = int(floor.replace('postauthor', ''))
                    if floor == 0:
                        reply = main
                    else: reply = post.xpath('.//span[@class="postcontent ubbcode"]//text()')
                    date = post.xpath('.//div[@class="postInfo"]/span//text()')
                                        
                    if len(reply) != 0:
                        reply = clean_quote(reply)
                        reply = clean_reply(reply)
                    if floor % 1000 == 0:
                        reply = clean_s(reply)

                    s.append({'uid': int(uid.split('=')[-1]), 
                            'floor': floor, 
                            'reply': reply,
                            'date': date})

                with open(f'{file}//{i}.json', 'a+', encoding='utf-8') as f:
                    json.dump(s, f, ensure_ascii=False)
            
            else:
                return 103
            time.sleep(0.25)
        return 1

    def loc_floor(self, tid: str, floor: int) -> int:
        try:
            r = requests.get(f'https://bbs.nga.cn/read.php?tid={tid}', headers=self.headers)
        except Exception:
            return 101
        try:
            total = int(re.search(r"',\d+:\d+", r.text).group().split(':')[1])
        except Exception:
            total = 1
        
        page = floor // 20 + 1
        if page == 1:
            main = etree.HTML(r.text).xpath('//*[@id="postcontent0"]/text()')
        if page > total:
            return 404

        try:
            r = requests.get(f'https://bbs.nga.cn/read.php?tid={tid}&page={page}', headers=self.headers)
        except Exception:
            return 102
        if r.status_code == 200:
            data = etree.HTML(r.text)
            post = data.xpath('//table[@class="forumbox postbox"]')[floor % 20]
            uid = post.xpath('.//a[@class="author b"]/@*')[0].split('=')[-1]
            if floor == 0:
                reply = main
            else: reply = post.xpath('//*[@id="postcontent' + str(floor) + '"]//text()')
            # print(reply[0])

            if len(reply) != 0:
                reply = clean_quote(reply)
                reply = clean_reply(reply)
            if floor % 1000 == 0:
                reply = clean_s(reply)

            print(uid, reply)
        
        return 1


if __name__ == '__main__':
    # HB_tid: 28817641, 29365342, 29593236, 29799537, 29874047, 30226559
    parser = argparse.ArgumentParser()
    parser.add_argument('--tid', type=str, default='0', help='tid')
    parser.add_argument('--p', type=int, default=1, help='start page')
    parser.add_argument('--loc', type=int, default=-1, help='locate floor reply')
    args = parser.parse_args()
    if args.loc < 0:
        print(NGA().get_reply(args.tid, args.p))
    else: NGA().loc_floor(args.tid, args.loc)

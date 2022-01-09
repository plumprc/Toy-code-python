import re
import os
import time
import json
import requests
from tqdm import tqdm
from lxml import etree
import argparse

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

    def get_reply(self, tid: str, start: int, file: str = 'merge') -> int:
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
                    reply = post.xpath('.//span[@class="postcontent ubbcode"]//text()')
                    date = post.xpath('.//div[@class="postInfo"]/span//text()')
                    s.append({'uid': int(uid.split('=')[-1]), 
                            'floor': int(floor.replace('postauthor', '')), 
                            'reply': main if floor == 'postauthor0' else reply,
                            'date': date})

                with open(f'{file}//{i}.json', 'a+', encoding='utf-8') as f:
                    json.dump(s, f, ensure_ascii=False)
            
            else:
                return 103
            time.sleep(0.25)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tid', type=str, default='0', help='tid')
    parser.add_argument('--p', type=int, default=1, help='p')
    args = parser.parse_args()
    print(NGA().get_reply(args.tid, args.p))

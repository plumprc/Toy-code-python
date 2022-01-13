import json
import re
import sys
from wordcloud import WordCloud
import jieba
import jieba.analyse
from matplotlib import pyplot as plt
from matplotlib import colors

color = ["mediumseagreen", "violet", "deeppink", "dodgerblue", "darkorange"]
colormap = colors.ListedColormap(color)
un_seg = ['妮老师', '牛子豪', '猫虚', '哈哈哈哈', '我不好说', '巴旦木', '普罗维登', 
        '捏麻麻滴', '捏麻麻地', '一号楼', '二号楼', '三号楼', '四号楼', '五号楼', 
        '量子二踢腿', '瓶gachi', '季gachi', '妮gachi', '单推人', '露露耶', '露露子']

def stop_wordlist():
    stopwords = [line.strip() for line in open('stop.txt',encoding='utf-8').readlines()]
    return stopwords

def create_wordcloud(filename):
    f = open(filename + '.txt', encoding='utf-8').read()
    for seg in un_seg:
        jieba.suggest_freq((seg), True)

    wordlist = jieba.analyse.extract_tags(f, topK=100)
    cloud = ''
    stopwords = stop_wordlist()
    for word in wordlist:
        if word not in stopwords:
            if word != '\t':
                cloud += word
                cloud += " "
    del(wordlist)
    wordcloud = WordCloud(
        background_color="white",
        max_words=2000,
        font_path='C:\Windows\Fonts\simfang.ttf',
        height=400,
        width=600,
        max_font_size=300,
        random_state=100,
        colormap=colormap
    )
    word = wordcloud.generate(cloud)
    plt.imshow(word)
    plt.axis("off")
    plt.savefig(filename + '.png', dpi=1200, bbox_inches='tight')

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
            r = re.findall(r'\[color=\S+?\]', s)
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
            s = s.replace('[/url]', '')
            s = s.replace('[/dice]', '')

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
                r = re.findall(r'\[color=\S+?\]', s)
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
                s = s.replace('[/url]', '')
                s = s.replace('[/dice]', '')

                with open(str(uid) + '.txt', 'a+', encoding='utf-8') as f:
                    f.write(s + '\n')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("please input file or uid")
    elif len(sys.argv) == 2:
        filename = 'statistics/' + sys.argv[1]
        generate_context(filename)
        create_wordcloud(filename)
    else:
        filename = 'statistics/' + sys.argv[1]
        uid = sys.argv[2]
        generate_context_by_uid(filename, uid)
        create_wordcloud(uid)

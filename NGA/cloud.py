import sys
from wordcloud import WordCloud
import jieba
import jieba.analyse
from matplotlib import pyplot as plt

def stop_wordlist():
    stopwords = [line.strip() for line in open('stop.txt',encoding='utf-8').readlines()]
    return stopwords

def create_wordcloud(filename):
    f = open(filename + '.txt', encoding='utf-8').read()
    jieba.suggest_freq(('妮老师'), True)
    jieba.suggest_freq(('牛子豪'), True)
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
        random_state=100
    )
    word = wordcloud.generate(cloud)
    plt.imshow(word)
    plt.axis("off")
    plt.savefig(filename + '.png', dpi=1200, bbox_inches='tight')

if __name__ == '__main__':
    create_wordcloud(sys.argv[1])

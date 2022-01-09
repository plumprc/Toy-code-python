import sys
from wordcloud import WordCloud
import jieba
import jieba.analyse
from matplotlib import pyplot as plt

def create_wordcloud(filename):
    f = open(filename + '.txt', encoding='utf-8').read()
    wordlist = jieba.analyse.extract_tags(f, topK=40)
    wordlist = " ".join(wordlist)
    print(type(wordlist))
    wordcloud = WordCloud(
        background_color="white",
        max_words=2000,
        font_path='C:\Windows\Fonts\simfang.ttf',
        height=600,
        width=1000,
        max_font_size=300,
        random_state=30
    )
    word = wordcloud.generate(wordlist)
    plt.imshow(word)
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    print(type(sys.argv[1]))
    create_wordcloud(sys.argv[1])

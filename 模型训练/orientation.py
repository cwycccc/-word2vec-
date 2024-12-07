# coding:utf-8
import re
import gensim
import jieba
import jieba.analyse

print('主程序开始...')

print('读取word2vec...')
word2vec = gensim.models.Word2Vec.load('wiki.model')  # 加载预训练的word2vec模型
print('读取word2vec结束！')

print('读取停用词...')
stopwords = []
for word in open('stopwords.txt', 'r', encoding='utf-8'):
    if word.strip():
        stopwords.append(word.strip())
jieba.analyse.set_stop_words('stopwords.txt')  # 设置停用词
print('停用词读取结束！')

print('读取正向词...')
positives = []
for word in open('positive.txt', 'r', encoding='utf-8'):
    if word.strip():
        positives.append(word.strip())  # 读取正向词
print('正向词读取结束！')

print('读取负向词...')
negatives = []
for word in open('negative.txt', 'r', encoding='utf-8'):
    if word.strip():
        negatives.append(word.strip())  # 读取负向词
print('负向词读取结束！')

# 加载小说文本
print('读取小说文本...')
with open('novel.txt', 'r', encoding='utf-8') as f:
    novel_text = f.read()
print('小说文本读取结束！')

# 预处理：分句处理
print('分句处理...')
contents = re.split(u'。|\n', novel_text)  # 按句号或换行分割
contents = filter(lambda s: (s if s != '' else None), contents)
contents = list(contents)
print('分句处理结束！')

# 停用词去除
print('停用词去除...')
results = []
for content in contents:
    seg = jieba.cut(content)  # 对每个句子进行分词
    results.append([c for c in seg if c.strip() != '' and c not in stopwords])  # 去掉停用词
contents = results
print('停用词去除结束！')

# 关键词抽取
print('关键词抽取...')
keywords = jieba.analyse.textrank(novel_text, topK=10, withWeight=True, allowPOS=('nt', 'ns', 'n', 'vn', 'v'))
print('关键词抽取结束！')

# 情感计算：计算句子的情感得分
print('情感计算...')
orientations = []
for content in contents:
    pos_num = 0
    pos_value = 0
    neg_num = 0
    neg_value = 0
    for word in content:
        pos = 0
        neg = 0
        # 计算与正向词的相似度
        for positive in positives:
            try:
                temp = word2vec.similarity(word, positive)
                if temp > pos:
                    pos = temp
            except:
                continue
        # 计算与负向词的相似度
        for negative in negatives:
            try:
                temp = word2vec.similarity(word, negative)
                if temp > neg:
                    neg = temp
            except:
                continue
        if pos > neg and pos > 0.5:  # 如果正向情感得分更高
            pos_num += 1
            pos_value += pos
        if neg > pos and neg > 0.5:  # 如果负向情感得分更高
            neg_num += 1
            neg_value += neg
    pos_ave = (pos_value / pos_num) if pos_num != 0 else 0
    neg_ave = (neg_value / neg_num) if neg_num != 0 else 0
    orientations.append(pos_ave if pos_ave > neg_ave else -neg_ave)
print('情感计算结束！')

# 计算情感变化趋势
print('情感变化趋势...')
pos_num = 0
pos_sum = 0
neg_num = 0
neg_sum = 0
for orientation in orientations:
    if orientation > 0:
        pos_num += 1
        pos_sum += orientation
    else:
        neg_num += 1
        neg_sum += orientation
pos_orientation = pos_sum / pos_num if pos_num != 0 else 0
neg_orientation = neg_sum / neg_num if neg_num != 0 else 0

# 输出情感倾向性
print('%lf' % (pos_orientation if pos_orientation > -neg_orientation else neg_orientation))
print('情感分析结束！')

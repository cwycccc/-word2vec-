import jieba
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 词向量模型加载（假设你已经训练好模型或者下载了预训练模型）
model = Word2Vec.load("wiki.model")  # 替换为你训练的模型文件

# 示例句子
s1 = "苏州有多条公路正在施工，造成局部地区汽车行驶非常缓慢。"
s2 = "苏州最近有多条公路在施工，导致部分地区交通拥堵，汽车难以通行。"
s3 = "苏州是一座美丽的城市，四季分明，雨量充沛。"

# 分词函数
def segment(text):
    return list(jieba.cut(text))

# 计算句子的向量表示（通过计算词向量的平均值）
def get_sentence_vector(sentence, model):
    words = segment(sentence)
    word_vectors = []
    for word in words:
        if word in model.wv:  # 如果词汇在词向量模型中
            word_vectors.append(model.wv[word])
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # 返回词向量的平均值
    else:
        return np.zeros(model.vector_size)  # 如果没有词向量，返回零向量

# 计算欧几里得距离
def euclidean_distance(sen1, sen2, model):
    vec1 = get_sentence_vector(sen1, model)
    vec2 = get_sentence_vector(sen2, model)
    distance = np.linalg.norm(vec1 - vec2)  # 欧几里得距离
    return distance

# 计算并输出欧几里得距离
print("==== 句子欧几里得距离 ====")
print(f"s1 | s1: {euclidean_distance(s1, s1, model):.4f}")
print(f"s1 | s2: {euclidean_distance(s1, s2, model):.4f}")
print(f"s1 | s3: {euclidean_distance(s1, s3, model):.4f}")

# 带权重的句子欧几里得距离
def get_weighted_sentence_vector(sentence, model, weight_func=None):
    words = segment(sentence)
    word_vectors = []
    for word in words:
        if word in model.wv:  # 如果词汇在词向量模型中
            weight = weight_func(word) if weight_func else 1
            word_vectors.append(model.wv[word] * weight)
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # 返回加权词向量的平均值
    else:
        return np.zeros(model.vector_size)  # 如果没有词向量，返回零向量



# 带权重的欧几里得距离
def weighted_euclidean_distance(sen1, sen2, model, weight_func):
    vec1 = get_weighted_sentence_vector(sen1, model, weight_func)
    vec2 = get_weighted_sentence_vector(sen2, model, weight_func)
    distance = np.linalg.norm(vec1 - vec2)  # 欧几里得距离
    return distance


# 计算两个句子的相似度
def sentence_similarity(sen1, sen2, model):
    vec1 = get_sentence_vector(sen1, model)
    vec2 = get_sentence_vector(sen2, model)
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return similarity

# 计算语义距离
def sentence_distance(sen1, sen2, model):
    similarity = sentence_similarity(sen1, sen2, model)
    distance = 1 - similarity  # 语义距离 = 1 - 相似度
    return distance

# 计算并输出相似度与语义距离
print("==== 句子相似度和语义距离 ====")
print(f"s1 | s1: 相似度 = {sentence_similarity(s1, s1, model):.4f}, 语义距离 = {sentence_distance(s1, s1, model):.4f}")
print(f"s1 | s2: 相似度 = {sentence_similarity(s1, s2, model):.4f}, 语义距离 = {sentence_distance(s1, s2, model):.4f}")
print(f"s1 | s3: 相似度 = {sentence_similarity(s1, s3, model):.4f}, 语义距离 = {sentence_distance(s1, s3, model):.4f}")

# 权重设置（例如，名词和动词权重为1，其他为0.8）
def get_weighted_sentence_vector(sentence, model, weight_func=None):
    words = segment(sentence)
    word_vectors = []
    for word in words:
        if word in model.wv:  # 如果词汇在词向量模型中
            weight = weight_func(word) if weight_func else 1
            word_vectors.append(model.wv[word] * weight)
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # 返回加权词向量的平均值
    else:
        return np.zeros(model.vector_size)  # 如果没有词向量，返回零向量

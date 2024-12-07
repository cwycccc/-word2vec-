import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from concurrent.futures import ThreadPoolExecutor
import re
from scipy.stats import gmean
from collections import Counter


# 设定路径和窗口大小
input_folder = "filtered"  # 输入文件夹路径
output_folder = "windowscev"  # 输出文件夹路径
window_size = 125  # 每个窗口包含的词数
model_file = "wiki.model"  # 已经训练好的模型文件

# 加载模型
w2v_model = Word2Vec.load(model_file)

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def sanitize_filename(filename):
    """将文件名中的特殊字符替换为下划线"""
    return re.sub(r'[^\w\-_\. ]', '_', filename)


def rename_files_in_folder(folder):
    """递归遍历文件夹并修复文件名"""
    for root, dirs, files in os.walk(folder):
        for file in files:
            sanitized = sanitize_filename(file)
            if file != sanitized:
                os.rename(os.path.join(root, file), os.path.join(root, sanitized))
                print(f"Renamed: {file} -> {sanitized}")


# 修复文件名
rename_files_in_folder(input_folder)


def check_vector_variability(window_vectors):
    """检查窗口向量的标准差，查看它们是否有多样性"""
    if len(window_vectors) < 2:
        return False
    stddev = np.std(window_vectors, axis=0)
    print(f"窗口向量的标准差: {stddev[:10]}")  # 打印前10个维度的标准差
    return np.any(stddev > 1e-5)  # 如果标准差大于某个阈值，表示有差异


def check_covariance_matrix(window_vectors):
    """检查协方差矩阵及其特征值"""
    P = np.array(window_vectors)
    covariance_matrix = np.cov(P.T)
    print(f"协方差矩阵:\n{covariance_matrix}")
    
    # 计算特征值
    eigvals = np.linalg.eigvals(covariance_matrix)
    print(f"特征值: {eigvals}")
    return eigvals


def get_min_vol_ellipse(P: np.ndarray):
    """获取最小体积椭球"""
    n = P.shape[1]
    covariance_matrix = np.cov(P.T)
    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
    A = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return A, eigvals


def two_opt_swap(route, i, j):
    """交换路径中的两段"""
    new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
    return new_route


def two_opt(route, distance_matrix):
    """2-opt算法优化路径"""
    best_route = route
    best_distance = calculate_total_distance(best_route, distance_matrix)
    
    for i in range(1, len(route) - 1):
        for j in range(i + 1, len(route)):
            new_route = two_opt_swap(best_route, i, j)
            new_distance = calculate_total_distance(new_route, distance_matrix)
            if new_distance < best_distance:
                best_route = new_route
                best_distance = new_distance
    return best_route, best_distance


def calculate_total_distance(route, distance_matrix):
    """计算路径的总距离"""
    return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))


def calculate_distance_matrix(window_vectors):
    """计算窗口向量之间的欧几里得距离矩阵"""
    num_windows = len(window_vectors)
    distance_matrix = np.zeros((num_windows, num_windows))
    for i in range(num_windows):
        for j in range(i + 1, num_windows):
            dist = np.linalg.norm(window_vectors[i] - window_vectors[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix


def compute_window_vectors(sentences):
    """计算窗口平均向量"""
    window_vectors = []
    for sentence in sentences:
        for i in range(0, len(sentence), window_size):
            window = sentence[i:i + window_size]
            vectors = [w2v_model.wv[word] for word in window if word in w2v_model.wv]
            if vectors:
                window_embedding = np.mean(vectors, axis=0)
                window_vectors.append(window_embedding)
    return window_vectors


def get_volume(chunk_emb: list, tolerance: float = 0.01, emb_dim: int = 300) -> float:
    """计算窗口向量的体积"""
    P = np.array(chunk_emb)
    
    rank = np.linalg.matrix_rank(P, tol=tolerance)     
    if rank < emb_dim or (rank == emb_dim and P.shape[0] <= emb_dim):
        tempA = P[1:, :].T - P[0, :].reshape(-1, 1) @ np.ones((1, P.shape[0] - 1))
        U, S, _ = np.linalg.svd(tempA)
        S1 = U[:, :rank-1]
        tempP = np.vstack([(S1.T @ tempA).T, np.zeros((1, rank-1))])
        A, _ = get_min_vol_ellipse(tempP)
    else:
        A, _ = get_min_vol_ellipse(P)

    _, S, _ = np.linalg.svd(A)
    return 1 / gmean(np.sqrt(S))  # 使用几何均值计算体积


def check_word_distribution(sentences):
    """检查每个窗口中单词的分布"""
    word_counts = Counter(word for sentence in sentences for word in sentence)
    print(f"单词分布: {word_counts.most_common(10)}")  # 打印最常见的前10个单词


def process_file(input_file, txt_file, category):
    """处理单个文件"""
    print(f"正在处理文件: {txt_file} (类别: {category})")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip().split() for line in f]

        print(f"文件 {txt_file} 成功读取")
        
        # 检查单词分布
        check_word_distribution(sentences)
        
        # 计算窗口向量
        window_vectors = compute_window_vectors(sentences)
        
        print(f"文件 {txt_file} 窗口向量计算完成")
        
        # 检查窗口向量的多样性
        if not check_vector_variability(window_vectors):
            print(f"警告: {txt_file} 的窗口向量几乎相同")
        
        # 计算相邻窗口之间的欧几里得距离
        window_distances = [np.linalg.norm(window_vectors[i] - window_vectors[i-1]) for i in range(1, len(window_vectors))]
        
        # 计算语义速度
        semantic_speed = np.mean(window_distances) if window_distances else 0
        
        # 计算实际路径长度
        actual_path_length = sum(window_distances)
        
        # 计算迂回性使用2-opt算法
        if len(window_vectors) > 1:
            # 计算窗口向量之间的欧几里得距离矩阵
            distance_matrix = calculate_distance_matrix(window_vectors)
            
            # 初始化一个简单的路径（从0到N-1）
            initial_route = list(range(len(window_vectors)))
            
            # 使用2-opt算法优化路径
            optimized_route, optimized_distance = two_opt(initial_route, distance_matrix)
            
            # 计算最短路径长度
            shortest_path_length = optimized_distance
        else:
            shortest_path_length = 0
        
        # 计算迂回性
        if shortest_path_length > 0:
            circuitousness = actual_path_length / shortest_path_length
        else:
            circuitousness = np.inf  # 如果最短路径为0，迂回性为无穷大
        
        # 计算体积
        volume = get_volume(window_vectors)
        
        print(f"{txt_file} 的语义速度为: {semantic_speed}")
        print(f"{txt_file} 的语义迂回性为: {circuitousness}")
        print(f"{txt_file} 的体积为: {volume}")
        
        # 返回带有类别信息的结果
        return {
            'file': txt_file,
            'category': category,  # 新增类别字段
            'semantic_speed': semantic_speed,
            'circuitousness': circuitousness,
            'volume': volume
        }
    except Exception as e:
        print(f"处理文件 {txt_file} 时出现错误: {e}")
        return None


def process():
    # 获取所有类别文件夹
    categories = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]

    # 存储语义速度和迂回性结果
    results = []
    
    # 处理每个类别的文件夹
    for category in categories:
        category_folder = os.path.join(input_folder, category)
        txt_files = [f for f in os.listdir(category_folder) if f.endswith('.txt')]

        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(process_file, os.path.join(category_folder, file), file, category): file for file in txt_files}
            
            for future in future_to_file:
                file = future_to_file[future]
                result = future.result()
                if result is not None:
                    results.append(result)  # 将结果添加到结果列表中

    # 保存结果为DataFrame并输出
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_folder, "output_results.csv"), index=False)
    print("结果已保存")


if __name__ == "__main__":
    process()

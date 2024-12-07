import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: 加载数据
# 替换 'your_file_path.csv' 为实际的文件路径
data = pd.read_csv('output_results.csv',encoding='utf-8')

# Step 2: 检查数据
print("数据概览：")
print(data.head())
print(data.info())

# Step 3: 重命名列名（如果需要）
# 替换为你实际数据的列名
data.columns = [
    "file_name", "category", "semantic_speed", 
    "circuitousness", "volume", "rating", 
    "rating_count", "citations", "downloads"
]

# Step 4: 按类别拆分数据
papers = data[data["category"] == "论文"]  # 替换为数据中代表论文的分类
novels = data[data["category"] == "小说"]  # 替换为数据中代表小说的分类

# Step 5: 计算相关性
# 论文相关性
papers_corr = papers[["semantic_speed", "circuitousness", "volume", "citations", "downloads"]].corr()
print("论文相关性矩阵：\n", papers_corr)

# 小说相关性
novels_corr = novels[["semantic_speed", "circuitousness", "volume", "rating", "rating_count"]].corr()
print("小说相关性矩阵：\n", novels_corr)

# Step 6: 可视化相关性（热力图）
plt.figure(figsize=(10, 8))
sns.heatmap(papers_corr, annot=True, cmap='coolwarm')
plt.title('论文相关性热力图')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(novels_corr, annot=True, cmap='coolwarm')
plt.title('小说相关性热力图')
plt.show()

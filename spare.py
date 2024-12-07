# coding:utf-8
import jieba
import re
import os

# 定义输入文件夹和输出文件夹
input_folder = 'resources'  # 输入文件夹
output_folder = 'filtered'  # 输出文件夹
debug_output_folder = 'output'  # 调试输出文件夹（用于查看去除空格后的文本）

# 如果输出文件夹和调试输出文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(debug_output_folder):
    os.makedirs(debug_output_folder)

print('主程序执行开始...')

# 获取输入文件夹中的所有子文件夹（如"论文"和"小说"） 
subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

count = 1  # 用于记录处理的行数
cn_reg = '^[\u4e00-\u9fa5]+$'  # 匹配中文字符的正则表达式

# 加载停用词列表
stopwords = set()
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set(f.read().strip().split('\n'))

# 遍历子文件夹
for subfolder in subfolders:
    subfolder_path = os.path.join(input_folder, subfolder)  # 子文件夹路径
    sub_output_folder = os.path.join(output_folder, subfolder)  # 子文件夹对应的输出路径
    sub_debug_output_folder = os.path.join(debug_output_folder, subfolder)  # 子文件夹对应的调试输出路径

    # 如果子文件夹对应的输出路径不存在，则创建
    if not os.path.exists(sub_output_folder):
        os.makedirs(sub_output_folder)
    if not os.path.exists(sub_debug_output_folder):
        os.makedirs(sub_debug_output_folder)

    print(f'开始处理子文件夹: {subfolder}...')
    
    # 获取当前子文件夹中的所有 .txt 文件
    input_files = [f for f in os.listdir(subfolder_path) if f.endswith('.txt')]

    for input_file_name in input_files:
        input_file_path = os.path.join(subfolder_path, input_file_name)  # 输入文件路径
        output_file_name = os.path.join(sub_output_folder, f'filtered_{input_file_name}')  # 输出文件路径
        debug_output_file_name = os.path.join(sub_debug_output_folder, f'debug_{input_file_name}')  # 调试输出文件路径

        # 打开输入文件和输出文件
        with open(input_file_path, 'r', encoding='utf-8') as input_file, \
             open(output_file_name, 'w', encoding='utf-8') as output_file, \
             open(debug_output_file_name, 'w', encoding='utf-8') as debug_output_file:
            
            print(f"正在处理文件: {input_file_name}...")
            lines = input_file.readlines()  # 读取所有行
            all_filtered_words = []  # 用于存储所有分词结果
            
            for line in lines:
                # 去掉行首尾空白字符，包括行尾的空格和回车符
                clean_line = line.strip()  # 保证行尾空格被去除
                if clean_line:  # 跳过空行
                    # 去掉行内所有空格
                    clean_line_without_spaces = clean_line.replace(' ', '')  # 去掉所有空格
                    # 输出去掉空格后的文本到调试文件
                    debug_output_file.write(clean_line_without_spaces + '\n')
                    
                    # 使用 jieba 分词
                    segmented_line = jieba.cut(clean_line_without_spaces)
                    
                    # 过滤分词结果，仅保留符合正则的中文字符，并去除停用词
                    filtered_words = [word for word in segmented_line if re.search(cn_reg, word) and word not in stopwords]
                    
                    # 将过滤后的结果追加到总结果中
                    all_filtered_words.extend(filtered_words)
                
                count += 1
                if count % 10000 == 0:
                    print(f"目前已处理 {count} 条数据")

            # 将所有分词结果用空格隔开，并写入文件
            output_file.write(' '.join(all_filtered_words))  # 输出紧凑的内容，分词用空格隔开
            
print('分词和过滤程序执行结束！')
print('主程序执行结束！')

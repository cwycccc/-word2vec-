import re
import os
import jieba  # 导入jieba库

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

# 文本清理函数
def clean_text(text):
    """
    清理文本内容：
    - 去除HTML标签
    - 删除特殊符号
    - 去除标点符号（包括句号等）
    - 去除多余空格
    """
    # 删除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 删除特殊符号和标点，只保留中文字符、数字、字母
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    
    # 去除多余空格
    text = re.sub(r'\s+', '', text)
    
    return text


# 分词函数
def segment_text(text):
    """
    使用jieba对文本进行分词，并用空格分隔每个词。
    """
    return " ".join(jieba.lcut(text))

# 遍历每个子文件夹
for subfolder in subfolders:
    print(f'正在处理子文件夹：{subfolder}...')
    subfolder_path = os.path.join(input_folder, subfolder)
    output_subfolder_path = os.path.join(output_folder, subfolder)
    debug_subfolder_path = os.path.join(debug_output_folder, subfolder)

    # 如果输出文件夹中不存在该子文件夹，则创建
    if not os.path.exists(output_subfolder_path):
        os.makedirs(output_subfolder_path)
    if not os.path.exists(debug_subfolder_path):
        os.makedirs(debug_subfolder_path)

    # 遍历子文件夹中的所有文件
    for file_name in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, file_name)
        
        if os.path.isfile(file_path):
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 调试输出：去除多余空格后的文本
            debug_cleaned_text = re.sub(r'\s+', '', content)
            debug_file_path = os.path.join(debug_subfolder_path, file_name)
            with open(debug_file_path, 'w', encoding='utf-8') as debug_file:
                debug_file.write(debug_cleaned_text)
            
            # 清理文本
            cleaned_text = clean_text(content)
            
            # 分词
            segmented_text = segment_text(cleaned_text)
            
            # 保存分词后的文本到输出文件夹
            output_file_path = os.path.join(output_subfolder_path, file_name)
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(segmented_text)

print('所有文本处理完成！')

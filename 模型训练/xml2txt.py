# coding: utf-8
"""
这个代码是将从网络上下载的xml格式的wiki百科训练语料转为txt格式
wiki百科训练语料
wiki百科的中文语料库下载地址:
https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2 
"""

from gensim.corpora import WikiCorpus

if __name__ == '__main__':

    print('主程序开始...')

    input_file_name = 'zhwiki-latest-pages-articles.xml.bz2'
    output_file_name = 'wiki.cn.txt'
    print('开始读入wiki数据...')

    # 不再使用lemmatize参数，如果需要进行词形还原，可以在tokenizer_func中处理
    input_file = WikiCorpus(input_file_name, dictionary={})
    print('wiki数据读入完成！')
    output_file = open(output_file_name, 'w', encoding="utf-8")

    print('处理程序开始...')
    count = 0
    for text in input_file.get_texts():
        # 这里直接将文本以空格连接并写入文件
        output_file.write(' '.join(text) + '\n')
        count += 1
        if count % 10000 == 0:
            print(f'目前已处理 {count} 条数据')

    print('处理程序结束！')

    output_file.close()
    print('主程序结束！')

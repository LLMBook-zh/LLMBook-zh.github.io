import re
from collections import Counter

def extract_frequencies(texts):
    """
    将输入文本列表中的每个文本转换为带有结束标记'</w>'的单个字符序列，
    并计算每种序列的频率。此函数使用Counter来累加每种序列的出现次数，
    这样可以快速得到每个序列在文本中出现的总次数。
    
    参数:
    texts (list of str): 输入的字符串列表，每个字符串代表一段文本。
    
    返回:
    Counter: 一个计数器对象，键是字符串中的字符序列，值是该序列的频率。
    """
    tokens = Counter()
    for text in texts:
        text = ' '.join(text) + ' </w>'  # 将每个字符转换为带空格的形式，并在末尾添加'</w>'
        tokens.update(text.split())
    return tokens

def frequency_of_pairs(frequencies):
    """
    从给定的频率字典中计算所有相邻字符对的频率。通过遍历每个词元，
    查找并统计所有相邻字符对的出现频率。

    参数:
    frequencies (Counter): 词元到其频率的映射字典。

    返回:
    Counter: 字符对到其频率的映射字典。
    """
    pairs = Counter()
    for token, freq in frequencies.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    """
    合并词汇表中最频繁的字符对。此函数接受一个字符对和当前的词汇表，
    将所有包含该字符对的词元中的对应字符合并为一个单一字符，
    并更新词汇表以反映这一变化。

    参数:
    pair (tuple): 要合并的字符对。
    vocab (Counter): 当前的词元到频率的映射字典。

    返回:
    Counter: 更新后的词元到频率的映射字典。
    """
    bigram = re.escape(' '.join(pair))
    merged = ''.join(pair)
    new_vocab = Counter()
    for token in vocab:
        new_token = token.replace(bigram, merged)
        new_vocab[new_token] = vocab[token]
    return new_vocab

def encode_with_bpe(texts, num_merges):
    """
    使用字节对编码(BPE)算法对输入文本进行编码。此函数首先提取词元的初始频率，
    然后迭代合并频率最高的字符对，直到达到指定的合并次数或没有可合并的对为止。

    参数:
    texts (list of str): 输入的字符串列表，每个字符串代表一段文本。
    num_merges (int): 指定的最大合并次数。

    返回:
    Counter: 合并后的词元到频率的映射字典。
    """
    vocab = extract_frequencies(texts)
    for _ in range(num_merges):
        pairs = frequency_of_pairs(vocab)
        if not pairs:
            break
        most_frequent = pairs.most_common(1)[0][0]
        vocab = merge_vocab(most_frequent, vocab)
    return vocab

# 示例用法
num_merges = 1000
bpe_vocab = encode_with_bpe(data, num_merges)
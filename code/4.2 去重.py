import string
import re
from nltk.util import ngrams

class CleanerDedupLineByNgram():
    def __init__(self):
        # 定义行分隔符和元组分隔符
        self.line_delimiter = list("\n")
        chinese_punctuation = "，。！？：；“”‘’（）《》【】、|—"
        self.gram_delimiter = list(string.punctuation) + list(chinese_punctuation) + [' ']
    def clean_single_text(self, text: str, n: int = 5, thre_sim: float = 0.95) -> str:
        # 依靠行分隔符分割所有行
        lines = [each for each in re.split('|'.join(map(re.escape, self.line_delimiter)), text) if each != '']
        lineinfo, last = list(), {}
        for idx, line in enumerate(lines): # 计算每行的 n 元组
            # 依靠元组分隔符分割所有 n 元组，并将其暂时存储到lineinfo里
            grams = [each for each in re.split('|'.join(map(re.escape, self.gram_delimiter)), line) if each != '']
            computed_ngrams = list(ngrams(grams, min(len(grams), n)))
            lineinfo.append({
                "lineno": idx, "text": line, "n": min(len(grams), n), "ngrams": computed_ngrams, "keep": 0
            })
            
        for idx, each in enumerate(lineinfo): # 过滤掉和相邻行之间 n 元组的Jaccard相似度超过thre_sim的行
            if last == {}:
                each["keep"], last = 1, each
            else:
                # 计算相邻行间的Jaccard相似度
                ngrams_last, ngrams_cur = set(last["ngrams"]), set(each["ngrams"]) 
                ngrams_intersection, ngrams_union =  len(ngrams_last.intersection(ngrams_cur)), len(ngrams_last.union(ngrams_cur))
                jaccard_sim = ngrams_intersection / ngrams_union if ngrams_union != 0 else 0
                if jaccard_sim < thre_sim:
                    each["keep"], last = 1, each
        # 将所有未被过滤掉的 n 元组重新拼接起来
        text = self.line_delimiter[0].join([each["text"] for each in lineinfo if each["keep"] == 1])
        return text
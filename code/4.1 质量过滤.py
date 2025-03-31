from utils.evaluator import LangIdentifier

class FilterPassageByLangs():
    def __init__(self) -> None:
        # 使用LangIdentifier模块加载已经训练好的fastText模型
        self.language_identifier = LangIdentifier(model_path="utils/models/fasttext/lid.176.bin")
        self.reject_threshold = 0.5
    def filter_single_text(self, text: str, accept_lang_list: list) -> bool:
        # 使用fastText模型给text打分，每种语言生成一个置信分数
        labels, scores = self.language_identifier.evaluate_single_text(text)
        # 如果text所有语言的分数均比reject_threshold要低，则直接定义为未知语言
        if socres[0] < self.reject_thereshold:
            labels = ["uk"]
        accept_lang_list = [each.lower() for each in accept_lang_list]
        # 如果分数最高的语言标签不在配置文件期望的语言列表中，则丢弃该文本
        if labels[0] not in accept_lang_list:
            return True
        return False   
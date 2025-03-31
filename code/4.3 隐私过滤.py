from utils.rules.regex import REGEX_IDCARD
from utils.cleaner.cleaner_base import CleanerBase

class CleanerSubstitutePassageIDCard(CleanerBase):
    def __init__(self):
        super().__init__()
    def clean_single_text(self, text: str, repl_text: str = "**MASKED**IDCARD**") -> str:
        # 使用正则表达式REGEX_IDCARD匹配身份证号，用repl_text代替
        return self._sub_re(text=text, re_text=REGEX_IDCARD, repl_text=repl_text)
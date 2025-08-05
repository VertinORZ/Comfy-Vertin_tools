import re

class LanguageDetectorNode:
    """
    语言检测比较节点：检测输入文本主要语言并与选择的语言进行比较
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"multiline": True, "default": ""}),
                "target_language": (["Chinese", "English"], {"default": "Chinese"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("is_match",)
    FUNCTION = "detect_and_compare"
    CATEGORY = "Vertin工具"  # 已修改为Vertin工具分类

    def detect_language(self, text):
        """检测文本主要语言"""
        if not text.strip():
            return None  # 空文本返回None
        
        # 中文字符的Unicode范围
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        # 英文字符的Unicode范围
        english_chars = re.findall(r'[a-zA-Z]', text)
        
        # 计算中文字符和英文字符的数量
        chinese_count = len(chinese_chars)
        english_count = len(english_chars)
        
        # 如果没有检测到任何中文字符或英文字符，返回None
        if chinese_count == 0 and english_count == 0:
            return None
            
        # 根据字符数量判断主要语言
        if chinese_count >= english_count:
            return "Chinese"
        else:
            return "English"

    def detect_and_compare(self, text_input, target_language):
        """检测文本语言并与目标语言比较"""
        detected_lang = self.detect_language(text_input)
        
        # 如果无法检测语言，返回False
        if detected_lang is None:
            return (False,)
            
        # 比较检测到的语言与目标语言
        is_match = (detected_lang == target_language)
        return (is_match,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "LanguageDetectorNode": LanguageDetectorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LanguageDetectorNode": "语言检测"  # 建议使用中文显示名
}

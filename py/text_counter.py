# Comfy-Vertin_tools/py/text_counter.py
class TextCounter:
    """文本字符统计节点，统计输入文本的字符数量并输出原文本"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入文本": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    # 增加一个STRING类型的输出端口，用于返回原文本
    RETURN_TYPES = ("INT", "STRING")
    # 为两个输出端口命名
    RETURN_NAMES = ("字符数量", "文本内容")
    FUNCTION = "count_chars"
    CATEGORY = "Vertin工具"

    def count_chars(self, 输入文本):
        """统计输入文本的字符数量并返回原文本"""
        char_count = len(输入文本)
        # 同时返回字符数量和原文本内容
        return (char_count, 输入文本)

# 节点映射配置
NODE_CLASS_MAPPINGS = {
    "TextCounter": TextCounter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextCounter": "文本输入"
}

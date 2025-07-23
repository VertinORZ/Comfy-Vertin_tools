# 导入lora加密节点
from .lora_crypto_nodes import (
    NODE_CLASS_MAPPINGS as lora_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as lora_NODE_DISPLAY_NAME_MAPPINGS
)

# 导入图像标签保存节点（注意路径正确，若在py子目录需调整）
from .py.enhanced_image_tagger_save import (
    NODE_CLASS_MAPPINGS as tagger_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as tagger_NODE_DISPLAY_NAME_MAPPINGS
)

# 合并所有节点映射
NODE_CLASS_MAPPINGS = {
    **lora_NODE_CLASS_MAPPINGS,** tagger_NODE_CLASS_MAPPINGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **lora_NODE_DISPLAY_NAME_MAPPINGS,** tagger_NODE_DISPLAY_NAME_MAPPINGS
}

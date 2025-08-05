# 导入现有节点（保持不变）
from .lora_crypto_nodes import (
    NODE_CLASS_MAPPINGS as lora_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as lora_NODE_DISPLAY_NAME_MAPPINGS
)
from .py.enhanced_image_tagger_save import (
    NODE_CLASS_MAPPINGS as tagger_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as tagger_NODE_DISPLAY_NAME_MAPPINGS
)
from .py.text_counter import (
    NODE_CLASS_MAPPINGS as counter_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as counter_NODE_DISPLAY_NAME_MAPPINGS
)

# 导入语言检测比较节点
from .py.language_detector_node import (
    NODE_CLASS_MAPPINGS as langdetect_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as langdetect_NODE_DISPLAY_NAME_MAPPINGS
)

# 合并所有节点映射（添加新节点）
NODE_CLASS_MAPPINGS = {
    **lora_NODE_CLASS_MAPPINGS,** tagger_NODE_CLASS_MAPPINGS,
    **counter_NODE_CLASS_MAPPINGS,
    ** langdetect_NODE_CLASS_MAPPINGS  # 新增
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **lora_NODE_DISPLAY_NAME_MAPPINGS,** tagger_NODE_DISPLAY_NAME_MAPPINGS,
    **counter_NODE_DISPLAY_NAME_MAPPINGS,
    ** langdetect_NODE_DISPLAY_NAME_MAPPINGS  # 新增
}

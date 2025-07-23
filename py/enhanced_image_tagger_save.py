import os.path
import datetime
import torch
import numpy as np
import folder_paths
from PIL import Image


class EnhancedImageTaggerSave:
    """图像标签保存节点，采用00001_前缀_后缀的命名规则"""
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.name = "Enhanced Image Tagger Save"
        self.max_counter = 10000  # 计数器计数器最大数量限制
        self.start_counter = 1  # 计数器起始值

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"optional": True}),
                "文件输出路径": ("STRING", {"default": ""}),
                "文件名前缀": ("STRING", {"default": "comfyui"}),  # 可自定义的前缀
                "文件名后缀": ("STRING", {"default": "_R"}),    # 可自定义的后缀
                "时间戳": (["无", "秒级", "毫秒级"], {"default": "无"}),
                "图像格式": (["png", "jpg"], {"default": "png"}),
                "图像质量": ("INT", {"default": 80, "min": 10, "max": 100, "step": 1}),
                "预览图像": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "标签文本": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("STRING",)  # 增加输出端口，返回保存结果信息
    RETURN_NAMES = ("保存结果",)  # 输出端口名称
    FUNCTION = "save_data"
    OUTPUT_NODE = True
    CATEGORY = "Vertin工具"

    def save_data(self, 
                 图像=None, 
                 标签文本=None, 
                 文件输出路径="", 
                 文件名前缀="comfyui", 
                 文件名后缀="_R", 
                 时间戳="无", 
                 图像格式="png", 
                 图像质量=80, 
                 预览图像=True, 
                 prompt=None, 
                 extra_pnginfo=None):
        
        # 处理标签文本默认值
        if 标签文本 is None:
            标签文本 = ""
            
        # 检测输入内容
        has_image = 图像 is not None and (
            (isinstance(图像, list) and len(图像) > 0) or 
            (hasattr(图像, 'shape') and 图像.shape[0] > 0)
        )
        has_text = bool(标签文本.strip())
        
        # 验证输入
        if not has_image and not has_text:
            return {"ui": {"images": []}, "result": ("错误: 未提供图像或标签文本，无法保存",)}

        # 自动判断保存模式
        if has_image and has_text:
            save_mode = "两者都存"
        elif has_image:
            save_mode = "仅图像"
        else:
            save_mode = "仅文本"

        # 处理文本行
        text_lines = [line.strip() for line in 标签文本.split('\n') if line.strip()]
        if has_text and not text_lines:
            text_lines = [标签文本.strip()]
            
        total_images = 0
        if has_image:
            if isinstance(图像, list):
                total_images = len(图像)
            else:
                total_images = 图像.shape[0] if hasattr(图像, 'shape') else 0
                
        total_texts = len(text_lines) if has_text else 0
        
        # 同步文本与图像数量
        if save_mode == "两者都存":
            if total_texts < total_images:
                last_line = text_lines[-1] if text_lines else ""
                while len(text_lines) < total_images:
                    text_lines.append(last_line)
            elif total_texts > total_images:
                text_lines = text_lines[:total_images]
            total_texts = len(text_lines)
        
        # 确定处理数量
        total_items = 0
        if save_mode == "仅图像":
            total_items = total_images
        elif save_mode == "仅文本":
            total_items = total_texts
        else:
            total_items = total_images

        now = datetime.datetime.now()
        
        # 处理路径占位符
        文件输出路径 = 文件输出路径.replace("%date", now.strftime("%Y-%m-%d"))
        文件输出路径 = 文件输出路径.replace("%time", now.strftime("%H-%M-%S"))
        文件名前缀 = 文件名前缀.replace("%date", now.strftime("%Y-%m-%d"))
        文件名前缀 = 文件名前缀.replace("%time", now.strftime("%H-%M-%S"))
        文件名后缀 = 文件名后缀.replace("%date", now.strftime("%Y-%m-%d"))
        文件名后缀 = 文件名后缀.replace("%time", now.strftime("%H-%M-%S"))
        
        # 确认输出路径
        if 文件输出路径:
            if not os.path.exists(文件输出路径):
                try:
                    os.makedirs(文件输出路径, exist_ok=True)
                except Exception as e:
                    return {"ui": {"images": []}, "result": (f"错误: 无法创建路径 {文件输出路径} - {str(e)}",)}
            full_output_folder = os.path.normpath(文件输出路径)
        else:
            full_output_folder = folder_paths.get_output_directory()

        # 获取基础计数器（从1开始）
        base_counter = self.start_counter
        try:
            counter_key = f"{文件名前缀}_{文件名后缀}" if save_mode == "仅文本" else f"{文件名前缀}_{文件名后缀}"
            if save_mode == "仅文本":
                base_counter = self._get_text_file_counter(full_output_folder, 文件名前缀, 文件名后缀)
            else:
                _, _, temp_counter, _, _ = folder_paths.get_save_image_path(
                    counter_key, self.output_dir, 512, 512)
                base_counter = max(self.start_counter, temp_counter)
        except Exception as e:
            base_counter = self.start_counter

        # 检查计数器最大值
        if base_counter >= self.max_counter:
            base_counter = self.start_counter

        results = []
        saved_files = []  # 存储保存的文件信息

        # 批量处理
        for idx in range(total_items):
            # 计算当前文件的计数器（从1开始连续编号）
            current_base_counter = base_counter + idx
            if current_base_counter > self.max_counter:
                current_base_counter = self.start_counter + (current_base_counter - self.max_counter - 1)
                
            # 构建基础文件名 - 格式：00001_前缀_后缀
            prefix_part = f"{current_base_counter:05d}"  # 5位数字计数器
            base_file = f"{prefix_part}_{文件名前缀}"
            
            # 添加文件名后缀（如果有）
            if 文件名后缀:
                base_file = f"{base_file}_{文件名后缀}"
            
            # 添加时间戳（如果启用）
            if 时间戳 == "毫秒级":
                time_part = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
                base_file = f"{base_file}_{time_part}"
            elif 时间戳 == "秒级":
                time_part = now.strftime("%Y-%m-%d_%H-%M-%S")
                base_file = f"{base_file}_{time_part}"

            # 处理覆盖逻辑 - 现在默认不允许覆盖
            final_file = base_file
            current_counter = self.start_counter
            
            # 检查文件是否存在，如存在则递增计数器
            while True:
                # 格式：00001_前缀_后缀（如已有相同文件则递增）
                test_file = f"{current_counter:05d}_{文件名前缀}_{文件名后缀}" if current_counter > self.start_counter else base_file
                
                # 检查文件是否存在
                exists = False
                if save_mode in ["仅图像", "两者都存"]:
                    exists |= os.path.exists(os.path.join(full_output_folder, f"{test_file}.{图像格式}"))
                if save_mode in ["仅文本", "两者都存"]:
                    exists |= os.path.exists(os.path.join(full_output_folder, f"{test_file}.txt"))
                
                if exists:
                    current_counter += 1
                    if current_counter >= self.max_counter:
                        return {"ui": {"images": results}, "result": (f"错误: 计数器数已超过最大值 {self.max_counter}",)}
                else:
                    final_file = test_file
                    break

            # 保存图像
            if save_mode in ["仅图像", "两者都存"] and has_image and idx < total_images:
                try:
                    if isinstance(图像, list):
                        img_tensor = 图像[idx]
                    else:
                        img_tensor = 图像[idx]
                        
                    img_array = 255. * img_tensor.cpu().numpy()
                    img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
                    
                    image_path = os.path.join(full_output_folder, f"{final_file}.{图像格式}")
                    if 图像格式 == "png":
                        compress_level = max(0, min(9, (100 - 图像质量) // 10))
                        img.save(image_path, compress_level=compress_level)
                    else:
                        if img.mode == "RGBA":
                            img = img.convert("RGB")
                        img.save(image_path, quality=图像质量)
                    
                    saved_files.append(f"图像: {os.path.basename(image_path)}")
                    
                    if 预览图像:
                        results.append({
                            "filename": f"{final_file}.{图像格式}",
                            "subfolder": "",
                            "type": self.type
                        })
                except Exception as e:
                    saved_files.append(f"图像 {idx+1} 保存失败: {str(e)}")

            # 保存文本
            if save_mode in ["仅文本", "两者都存"] and has_text and idx < len(text_lines):
                try:
                    current_text = text_lines[idx]
                    text_path = os.path.join(full_output_folder, f"{final_file}.txt")
                    
                    os.makedirs(os.path.dirname(text_path), exist_ok=True)
                    
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(current_text)
                    
                    saved_files.append(f"文本: {os.path.basename(text_path)}")
                except Exception as e:
                    saved_files.append(f"文本 {idx+1} 保存失败: {str(e)}")

        # 构建输出结果文本
        if save_mode == "两者都存":
            result_text = f"成功保存 {total_items} 组图像和文本:\n"
        elif save_mode == "仅图像":
            result_text = f"成功保存 {total_items} 张图像:\n"
        else:
            result_text = f"成功保存 {total_items} 个文本文件:\n"
            
        result_text += "\n".join(saved_files)
        
        return {"ui": {"images": results}, "result": (result_text,)}
    
    def _get_text_file_counter(self, folder, prefix, suffix):
        """获取文本文件计数器（从1开始）"""
        if not os.path.exists(folder):
            return self.start_counter
            
        max_counter = self.start_counter - 1  # 初始化为0，确保至少返回1
        prefix_suffix = f"{prefix}_{suffix}" if suffix else prefix
        
        for filename in os.listdir(folder):
            if filename.endswith(".txt") and prefix_suffix in filename:
                try:
                    # 提取5位数字的计数器部分（文件名开头）
                    parts = filename.split('_')
                    if parts and parts[0].isdigit() and len(parts[0]) == 5:
                        num = int(parts[0])
                        if num > max_counter:
                            max_counter = num
                except:
                    continue
                    
        return max_counter + 1


# 节点注册
NODE_CLASS_MAPPINGS = {
    "EnhancedImageTaggerSave": EnhancedImageTaggerSave
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedImageTaggerSave": "图像标签批量保存工具"
}
    

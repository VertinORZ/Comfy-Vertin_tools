import os
import sys
import time
import torch
import logging
import tempfile
import base64
import atexit
from typing import List, Optional, Any, Dict, Tuple, Union
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# 确保comfy模块在路径中
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if comfy_path not in sys.path:
    sys.path.append(comfy_path)

# 直接导入模块
import folder_paths
# 为保持向后兼容性，仍然使用原来的导入方式
import comfy.sd as comfy_sd
import comfy.utils as comfy_utils

# 初始化日志
logger = logging.getLogger("VertinTools")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 全局列表跟踪需要删除的临时文件（仅用于紧急情况）
temp_files_to_clean: List[str] = []

# 程序退出时清理临时文件的函数（作为备用机制）
def clean_temp_files():
    """程序退出时清理可能残留的临时文件"""
    for temp_path in temp_files_to_clean:
        if not os.path.exists(temp_path):
            continue
            
        # 最多重试5次删除
        for attempt in range(5):
            try:
                os.unlink(temp_path)
                break
            except PermissionError:
                if attempt < 4:
                    time.sleep(attempt + 1)
            except:
                break
    temp_files_to_clean.clear()

# 注册退出时的清理函数（备用机制）
atexit.register(clean_temp_files)

# 加密标记和常量定义
ENCRYPTION_MARKER = b"VERTIN_ENCRYPTED"

"""加密相关工具函数"""
def is_encrypted_file(file_path):
    """检查文件是否为加密文件"""
    try:
        with open(file_path, "rb") as f:
            header = f.read(len(ENCRYPTION_MARKER))
            return header == ENCRYPTION_MARKER
    except Exception as e:
        logger.debug(f"检查文件 {file_path} 加密状态出错: {str(e)}")
        return False

def generate_key(password: str, salt: bytes) -> tuple[bytes, bytes]:
    """从密码和盐生成加密密钥"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt

def sanitize_folder_name(name: str) -> str:
    """清理文件夹名称中的不安全字符"""
    import re
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def encrypt_single_file(input_path, password, output_suffix, overwrite=False, skip_verification=True, use_password_folder=False):
    """加密单个文件的实现，使用下划线连接后缀名"""
    start_time = time.time()
    lora_filename = os.path.basename(input_path)
    try:
        if not os.path.exists(input_path):
            return f"失败: 文件不存在 {lora_filename}"
        
        if is_encrypted_file(input_path):
            return f"跳过: {lora_filename} 已加密"
        
        dir_name = os.path.dirname(input_path)
        base_name = os.path.splitext(lora_filename)[0]
        ext = os.path.splitext(lora_filename)[1]
        
        # 处理密码文件夹
        if use_password_folder:
            password_folder = sanitize_folder_name(password)
            output_dir = os.path.join(dir_name, password_folder)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = dir_name
        
        # 使用下划线连接后缀名
        suffix = output_suffix if output_suffix else "enc"
        output_filename = f"{base_name}_{suffix}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(output_path) and not overwrite:
            return f"跳过: {output_filename} 已存在（启用覆盖可替换）"
        
        # 读取并加密文件内容
        with open(input_path, "rb") as f:
            data = f.read()
        
        salt = os.urandom(16)
        key, _ = generate_key(password, salt)
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data)
        
        # 写入加密文件
        with open(output_path, "wb") as f:
            f.write(ENCRYPTION_MARKER + b"|||" + salt + b"|||" + encrypted_data)
        
        # 验证加密结果
        if not skip_verification:
            try:
                with open(output_path, "rb") as f:
                    verify_data = f.read()
                v_parts = verify_data.split(b"|||", 2)
                if len(v_parts) != 3 or v_parts[0] != ENCRYPTION_MARKER:
                    raise ValueError("加密文件格式验证失败")
                v_key, _ = generate_key(password, v_parts[1])
                Fernet(v_key).decrypt(v_parts[2])
            except Exception as e:
                os.remove(output_path)
                return f"失败: 加密验证失败 {lora_filename} - {str(e)}"
        
        duration = time.time() - start_time
        location_info = f"{output_dir}" if use_password_folder else f"{dir_name}"
        return f"成功: {lora_filename} → {output_filename}（保存在{location_info}） (耗时: {duration:.2f}秒)"
    
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"失败: {lora_filename} - {str(e)} (耗时: {duration:.2f}秒)"
        logger.error(error_msg)
        return error_msg

"""递归获取所有LoRA文件"""
def get_all_lora_files(include_none=True):
    lora_files = []
    valid_extensions = ('.safetensors', '.ckpt', '.bin', '.pth', '.pt', '.lora')
    lora_dirs = folder_paths.get_folder_paths("loras") or []
    
    if include_none:
        lora_files.append("None")
    
    if not lora_dirs:
        default_lora_dir = os.path.join(folder_paths.models_dir, "loras")
        if os.path.isdir(default_lora_dir):
            lora_dirs = [default_lora_dir]
    
    def scan_directory(directory, base_dir):
        if not os.path.isdir(directory):
            return
            
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                if os.path.isdir(item_path):
                    scan_directory(item_path, base_dir)
                elif os.path.isfile(item_path):
                    if (item.lower().endswith(valid_extensions) or 
                        is_encrypted_file(item_path)):
                        relative_path = os.path.relpath(item_path, base_dir)
                        lora_files.append(relative_path)
                        logger.debug(f"发现LoRA文件: {relative_path}")
        except PermissionError:
            logger.warning(f"没有权限访问目录: {directory}")
        except Exception as e:
            logger.error(f"扫描目录 {directory} 时出错: {str(e)}")
    
    for root_dir in lora_dirs:
        if os.path.isdir(root_dir):
            scan_directory(root_dir, root_dir)
    
    return sorted(list(set(lora_files)))

"""应用LoRA权重到模型"""
# 由于我们使用ComfyUI原生的LoRA加载机制，这个函数不再需要
def apply_lora(model, lora_data, strength):
    model_sd = model.state_dict()
    original_model = model
    model_keys = list(model_sd.keys())
    
    # 提取LoRA参数对
    lora_pairs = {}
    for name in lora_data:
        if name.endswith(".lora_down.weight"):
            base_name = name[:-len(".lora_down.weight")]
            up_name = f"{base_name}.lora_up.weight"
            if up_name in lora_data:
                lora_pairs[base_name] = (name, up_name)
        # 同时支持Qwen模型的LoRA格式
        elif name.endswith(".lora_down.weight"):
            base_name = name[:-len(".lora_down.weight")]
            up_name = f"{base_name}.lora_up.weight"
            alpha_name = f"{base_name}.alpha"
            if up_name in lora_data:
                lora_pairs[base_name] = (name, up_name, alpha_name if alpha_name in lora_data else None)
    
    logger.debug(f"发现 {len(lora_pairs)} 对LoRA参数")
    
    # 跟踪匹配到的参数
    matched_count = 0
    applied_count = 0
    
    # 尝试匹配并应用LoRA
    for base_name, lora_info in lora_pairs.items():
        down_name, up_name = lora_info[0], lora_info[1]
        alpha_name = lora_info[2] if len(lora_info) > 2 else None
        
        # 构建可能的匹配模式，包括Qwen模型的特殊格式
        possible_matches = [
            base_name,
            base_name.replace("transformer.", ""),
            f"model.{base_name}",
            base_name.replace(".", "_"),
            # Qwen模型特殊处理
            base_name.replace("lora_unet_", "transformer."),
            base_name.replace("lora_unet_", ""),
        ]
        
        # 如果是Qwen模型的特定格式，添加更多匹配模式
        if base_name.startswith("lora_unet_double_blocks_") or base_name.startswith("lora_unet_single_blocks_"):
            # 移除前缀并尝试匹配
            qwen_key = base_name.replace("lora_unet_", "")
            possible_matches.append(f"transformer.{qwen_key}")
            possible_matches.append(qwen_key)
            # 尝试将下划线替换为点
            dotted_key = qwen_key.replace("_", ".")
            possible_matches.append(f"transformer.{dotted_key}")
            possible_matches.append(dotted_key)
        
        matched_key = None
        for candidate in possible_matches:
            if candidate in model_keys:
                matched_key = candidate
                break
        
        if not matched_key:
            logger.debug(f"未找到匹配的模型参数: {base_name} (尝试了多种匹配方式)")
            continue
        
        # 找到匹配，应用LoRA
        matched_count += 1
        lora_down = lora_data[down_name]
        lora_up = lora_data[up_name]
        lora_alpha = lora_data.get(alpha_name, None) if alpha_name else None
        model_param = model_sd[matched_key]
        
        try:
            with torch.no_grad():
                if len(lora_down.shape) == 2 and len(lora_up.shape) == 2:
                    if lora_up.shape[1] == lora_down.shape[0]:
                        # 计算alpha缩放因子
                        if lora_alpha is not None:
                            scale = lora_alpha.item() / lora_down.shape[0]
                        else:
                            scale = 1.0
                        lora_weight = torch.matmul(lora_up, lora_down) * strength * scale
                    elif lora_down.shape[1] == lora_up.shape[0]:
                        # 计算alpha缩放因子
                        if lora_alpha is not None:
                            scale = lora_alpha.item() / lora_down.shape[1]
                        else:
                            scale = 1.0
                        lora_weight = torch.matmul(lora_down, lora_up) * strength * scale
                    else:
                        logger.warning(f"LoRA矩阵形状不兼容: {lora_down.shape} 和 {lora_up.shape}")
                        continue
                else:
                    logger.warning(f"不支持的LoRA参数形状: {lora_down.shape}")
                    continue
                
                if lora_weight.shape == model_param.shape:
                    model_sd[matched_key] += lora_weight
                    applied_count += 1
                    logger.debug(f"成功应用LoRA到 {matched_key} (强度: {strength})")
                else:
                    lora_weight_reshaped = lora_weight.reshape(model_param.shape)
                    if lora_weight_reshaped.shape == model_param.shape:
                        model_sd[matched_key] += lora_weight_reshaped
                        applied_count += 1
                        logger.debug(f"重塑后应用LoRA到 {matched_key}")
                    else:
                        logger.warning(f"LoRA权重形状不匹配模型参数: {lora_weight.shape} vs {model_param.shape}")
        except Exception as e:
            logger.error(f"应用LoRA到 {matched_key} 时出错: {str(e)}")
            continue
    
    # 加载修改后的状态字典
    try:
        original_model.load_state_dict(model_sd, strict=False)
        logger.debug(f"LoRA应用完成 - 匹配 {matched_count} 个参数，成功应用 {applied_count} 个")
        
        if hasattr(model, 'model'):
            model.model = original_model
            logger.debug("已更新ModelPatcher中的模型")
            
        return model
    except Exception as e:
        logger.error(f"加载修改后的状态字典时出错: {str(e)}")
        return model

# 获取LoRA文件的绝对路径
def get_lora_absolute_path(lora_name):
    lora_dirs = folder_paths.get_folder_paths("loras") or []
    
    if not lora_dirs:
        default_lora_dir = os.path.join(folder_paths.models_dir, "loras")
        if os.path.isdir(default_lora_dir):
            lora_dirs = [default_lora_dir]
    
    for lora_dir in lora_dirs:
        full_path = os.path.join(lora_dir, lora_name)
        if os.path.exists(full_path):
            return full_path
    
    return None

# 检查safetensors是否可用
try:
    from safetensors.torch import load_file as load_safetensors
    from safetensors.torch import save_file as save_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# LoRA加密器节点
class LoraEncryptor:
    def __init__(self):
        self.lora_dirs = folder_paths.get_folder_paths("loras") or []

    @classmethod
    def INPUT_TYPES(s):
        all_lora_files = get_all_lora_files(include_none=False)
        return {
            "required": {
                "lora文件名": (all_lora_files,),
                "加密密码": ("STRING", {
                    "default": "", "placeholder": "加密密码", "password": True
                }),
                "输出文件后缀": ("STRING", {
                    "default": "enc", "placeholder": "例如：enc"
                }),
            },
            "optional": {
                "覆盖已存在文件": ("BOOLEAN", {"default": False}),
                "跳过验证": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("加密结果",)
    FUNCTION = "encrypt_lora"
    CATEGORY = "Vertin工具"

    def encrypt_lora(self, lora文件名, 加密密码, 输出文件后缀, 覆盖已存在文件=False, 跳过验证=True):
        if not self.lora_dirs and not os.path.isdir(os.path.join(folder_paths.models_dir, "loras")):
            return ("错误: 未配置LoRA目录",)
        if not 加密密码:
            return ("错误: 请输入加密密码",)
            
        lora_path = get_lora_absolute_path(lora文件名)
        if not lora_path:
            return (f"错误: 未找到文件 {lora文件名}",)
            
        return (encrypt_single_file(lora_path, 加密密码, 输出文件后缀, 覆盖已存在文件, 跳过验证),)

# LoRA批量加密器节点
class LoraBatchEncryptor:
    def __init__(self):
        self.last_used_dirs = []  # 保存最近使用的目录

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "加密目录": ("STRING", {
                    "default": "", 
                    "placeholder": "输入或粘贴文件夹路径",
                    "tooltip": "可以输入任意文件夹的绝对路径"
                }),
                "加密密码": ("STRING", {
                    "default": "", "placeholder": "加密密码", "password": True
                }),
                "输出文件后缀": ("STRING", {
                    "default": "enc", "placeholder": "例如：enc"
                }),
            },
            "optional": {
                "覆盖已存在文件": ("BOOLEAN", {"default": False}),
                "跳过验证": ("BOOLEAN", {"default": True}),
                "包含子目录": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("加密结果汇总",)
    FUNCTION = "batch_encrypt"
    CATEGORY = "Vertin工具"

    def batch_encrypt(self, 加密目录, 加密密码, 输出文件后缀, 覆盖已存在文件=False, 跳过验证=True, 包含子目录=True):
        target_dir = 加密目录.strip()
            
        if not target_dir or not os.path.isdir(target_dir):
            return ("错误: 请输入有效的文件夹路径",)
        if not 加密密码:
            return ("错误: 请输入加密密码",)
            
        if target_dir not in self.last_used_dirs:
            self.last_used_dirs.insert(0, target_dir)
            if len(self.last_used_dirs) > 10:
                self.last_used_dirs = self.last_used_dirs[:10]
            
        results = []
        valid_extensions = ('.safetensors', '.ckpt', '.bin', '.pth', '.pt', '.lora')
        
        def scan_and_encrypt(directory):
            if not os.path.isdir(directory):
                return
                
            try:
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    
                    if os.path.isdir(item_path) and 包含子目录:
                        scan_and_encrypt(item_path)
                    elif os.path.isfile(item_path):
                        file_ext = os.path.splitext(item_path)[1].lower()
                        if (file_ext in valid_extensions and 
                            not is_encrypted_file(item_path)):
                            result = encrypt_single_file(
                                item_path, 加密密码, 输出文件后缀, 
                                覆盖已存在文件, 跳过验证,
                                use_password_folder=True
                            )
                            results.append(result)
            except PermissionError:
                results.append(f"警告: 没有权限访问目录 {directory}")
            except Exception as e:
                results.append(f"错误: 扫描目录 {directory} 时出错 - {str(e)}")
        
        scan_and_encrypt(target_dir)
        
        total = len(results)
        success = sum(1 for r in results if r.startswith("成功:"))
        skipped = sum(1 for r in results if r.startswith("跳过:"))
        failed = sum(1 for r in results if r.startswith("失败:"))
        
        sanitized_password = sanitize_folder_name(加密密码)
        summary = (f"批量加密完成 - 总计: {total}, 成功: {success}, "
                  f"跳过: {skipped}, 失败: {failed}\n所有加密文件均保存在原文件目录下的[{sanitized_password}]子文件夹\n\n详细结果:\n")
        summary += "\n".join(results)
        
        return (summary,)

# 带解密功能的LoRA加载器
class LoraDecryptLoader:
    """带解密功能的LoRA加载器，保留原生LoraLoader全部逻辑"""
    def __init__(self):
        self.loaded_lora = None  # 缓存 (路径, 数据, 密码)

    @classmethod
    def INPUT_TYPES(s):
        lora_files = get_all_lora_files(include_none=True)
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (lora_files, {"tooltip": "The name of the LoRA. Select 'None' to not load any LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            },
            "optional": {
                "password": ("STRING", {"default": "", "placeholder": "加密文件密码", "password": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "Vertin工具"
    DESCRIPTION = "支持加密LoRA文件的加载器，未加密文件可直接加载，选择'None'不加载任何LoRA"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, password=""):
        if lora_name == "None":
            return (model, clip)
            
        # 允许负值强度
        # if strength_model == 0 and strength_clip == 0:
        #     return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        
        # 检查缓存
        if self.loaded_lora is not None:
            cached_path, cached_lora, cached_password = self.loaded_lora
            if cached_path == lora_path and cached_password == password:
                lora = cached_lora
            else:
                self.loaded_lora = None

        if lora is None:
            lora = self._load_lora_data(lora_path, password)
            self.loaded_lora = (lora_path, lora, password)

        # 预处理Qwen模型的LoRA数据以提高兼容性
        lora = self._preprocess_qwen_lora(lora, model)
        
        # 临时抑制ComfyUI的LoRA警告日志（仅抑制特定的未加载键警告）
        lora_logger = logging.getLogger()  # 获取根日志记录器
        
        # 创建一个过滤器来只抑制特定的警告
        class LoraWarningFilter(logging.Filter):
            def filter(self, record):
                # 只过滤掉LoRA未加载键的警告
                if record.levelno == logging.WARNING and "lora key not loaded" in record.getMessage():
                    return False  # 过滤掉这条日志
                return True  # 允许其他日志通过
        
        # 添加过滤器
        warning_filter = LoraWarningFilter()
        for handler in lora_logger.handlers:
            handler.addFilter(warning_filter)
        
        try:
            # 使用最新的ComfyUI LoRA加载机制
            model_lora, clip_lora = comfy_sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        finally:
            # 移除过滤器
            for handler in lora_logger.handlers:
                handler.removeFilter(warning_filter)
            
        return (model_lora, clip_lora)

    def _preprocess_qwen_lora(self, lora_data, model):
        """预处理Qwen模型的LoRA数据以提高兼容性"""
        # 检查是否是Qwen模型
        try:
            # 检查模型类型
            is_qwen = hasattr(model, 'model') and hasattr(model.model, '__class__') and \
                      'QwenImage' in str(model.model.__class__)
        except:
            is_qwen = False
            
        if not is_qwen:
            # 检查键名中是否包含Qwen特有的模式
            qwen_patterns = ['double_blocks_', 'single_blocks_', 'img_attn_', 'img_mlp_', 'modulation.lin', 'blocks.']
            is_qwen = any(any(pattern in key for key in lora_data.keys()) for pattern in qwen_patterns)
        
        # 如果是Qwen模型，添加额外的键映射
        if is_qwen:
            processed_lora = lora_data.copy()
            keys_to_add = {}
            
            # 为Qwen模型创建额外的键映射
            for key in lora_data.keys():
                # 生成所有可能的键变体
                variants = self._generate_key_variants(key)
                
                # 为每个变体添加映射（如果不存在）
                for variant in variants:
                    if variant != key and variant not in processed_lora:
                        keys_to_add[variant] = lora_data[key]
                        # 将详细的键映射日志改为DEBUG级别
                        logger.debug(f"为Qwen模型添加键映射: {key} -> {variant}")
            
            # 合并新键
            processed_lora.update(keys_to_add)
            # 将统计信息日志改为DEBUG级别
            logger.debug(f"Qwen模型LoRA预处理完成，新增 {len(keys_to_add)} 个键映射")
            return processed_lora
            
        return lora_data

    def _generate_key_variants(self, key):
        """生成键的所有可能变体以提高兼容性"""
        variants = {key}  # 包含原始键
        
        # 添加或移除transformer.前缀
        if key.startswith("transformer."):
            variants.add(key[11:])  # 移除"transformer."前缀
        else:
            variants.add(f"transformer.{key}")  # 添加"transformer."前缀
            
        # 处理lora_unet_前缀
        if key.startswith("lora_unet_"):
            new_key = key[10:]  # 移除"lora_unet_"前缀
            variants.add(new_key)
            variants.add(f"transformer.{new_key}")
            # 将下划线替换为点
            dotted_key = new_key.replace("_", ".")
            variants.add(dotted_key)
            variants.add(f"transformer.{dotted_key}")
            
        # 处理下划线和点的转换
        if "_" in key and not key.startswith("lora_unet_"):
            dotted_key = key.replace("_", ".")
            variants.add(dotted_key)
            if not dotted_key.startswith("transformer."):
                variants.add(f"transformer.{dotted_key}")
                
        if "." in key and not key.startswith("lora_unet_"):
            underscored_key = key.replace(".", "_")
            variants.add(underscored_key)
            if not underscored_key.startswith("transformer."):
                variants.add(f"transformer.{underscored_key}")
                
        # 处理blocks模式
        if "blocks." in key:
            # 确保有各种前缀组合
            if not key.startswith("transformer."):
                variants.add(f"transformer.{key}")
                
        # 处理single和double blocks
        if "single.blocks." in key:
            variants.add(key.replace("single.blocks.", "single_blocks_"))
            variants.add(f"transformer.{key.replace('single.blocks.', 'single_blocks_')}")
        elif "single_blocks_" in key:
            variants.add(key.replace("single_blocks_", "single.blocks."))
            variants.add(f"transformer.{key.replace('single_blocks_', 'single.blocks.')}")
            
        if "double.blocks." in key:
            variants.add(key.replace("double.blocks.", "double_blocks_"))
            variants.add(f"transformer.{key.replace('double.blocks.', 'double_blocks_')}")
        elif "double_blocks_" in key:
            variants.add(key.replace("double_blocks_", "double.blocks."))
            variants.add(f"transformer.{key.replace('double_blocks_', 'double.blocks.')}")
            
        # 处理特定组件
        component_mappings = {
            "modulation.lin": "modulation_linear",
            "modulation_linear": "modulation.lin",
            "linear1": "linear_1",
            "linear_1": "linear1",
            "linear2": "linear_2",
            "linear_2": "linear2",
            "attn.qkv": "attn_qkv",
            "attn_qkv": "attn.qkv",
            "attn.proj": "attn_proj",
            "attn_proj": "attn.proj",
            "mlp.0": "mlp_0",
            "mlp_0": "mlp.0",
            "mlp.2": "mlp_2",
            "mlp_2": "mlp.2"
        }
        
        for old, new in component_mappings.items():
            if old in key:
                new_key = key.replace(old, new)
                variants.add(new_key)
                if not new_key.startswith("transformer."):
                    variants.add(f"transformer.{new_key}")
                if new_key.startswith("transformer."):
                    variants.add(new_key[11:])  # 移除transformer.前缀
        
        # 处理LoRA特定的键（.lora.down.weight, .lora.up.weight, .alpha）
        lora_patterns = [".lora.down.weight", ".lora.up.weight", ".alpha"]
        for pattern in lora_patterns:
            if pattern in key:
                # 确保有各种前缀组合
                base_key = key.replace(pattern, "")
                variants.add(f"{base_key}{pattern}")
                if not key.startswith("transformer."):
                    variants.add(f"transformer.{base_key}{pattern}")
                    
        # 处理更复杂的键结构
        # 例如: single.blocks.8.modulation.lin.lora.down.weight
        if "blocks." in key and ("lora" in key or "alpha" in key):
            # 分离基础键和LoRA部分
            parts = key.split(".")
            lora_part_indices = [i for i, part in enumerate(parts) if part in ["lora", "alpha"]]
            
            if lora_part_indices:
                # 为每个LoRA部分生成变体
                for i in lora_part_indices:
                    # 创建没有当前LoRA部分的键
                    base_parts = parts[:i] + parts[i+1:] if "alpha" in parts[i] else parts[:i]
                    base_key = ".".join(base_parts)
                    
                    # 为基本键生成变体
                    base_variants = {base_key}
                    if "_" in base_key:
                        base_variants.add(base_key.replace("_", "."))
                    if "." in base_key:
                        base_variants.add(base_key.replace(".", "_"))
                        
                    # 重新组合LoRA键
                    for base_variant in base_variants:
                        if "down.weight" in key:
                            variants.add(f"{base_variant}.lora.down.weight")
                            variants.add(f"transformer.{base_variant}.lora.down.weight")
                        elif "up.weight" in key:
                            variants.add(f"{base_variant}.lora.up.weight")
                            variants.add(f"transformer.{base_variant}.lora.up.weight")
                        elif "alpha" in key:
                            variants.add(f"{base_variant}.alpha")
                            variants.add(f"transformer.{base_variant}.alpha")
                            
        # 处理数字索引的变化
        import re
        # 查找键中的数字
        numbers = re.findall(r'\d+', key)
        for num in numbers:
            # 尝试不同的数字格式
            int_num = int(num)
            variants.add(key.replace(num, str(int_num)))  # 确保是整数格式
            
        return variants

    def _load_lora_data(self, lora_path: str, password: str):
        """加载LoRA数据，自动检测加密状态"""
        if is_encrypted_file(lora_path):
            if not password:
                raise ValueError(f"工作流已加密，请输入密码.请前往公众号“阿泰ATAI动态视觉”购买课程获取密码.")
            return self._decrypt_lora(lora_path, password)
        else:
            # 使用安全加载
            return comfy_utils.load_torch_file(lora_path, safe_load=True)

    def _decrypt_lora(self, lora_path: str, password: str):
        """解密LoRA文件并返回数据，加载后立即删除临时文件"""
        with open(lora_path, "rb") as f:
            data = f.read()

        parts = data.split(b"|||", 2)
        if len(parts) != 3 or parts[0] != ENCRYPTION_MARKER:
            raise ValueError("无效的加密LoRA文件格式")
        
        _, salt, encrypted_data = parts
        
        key, _ = generate_key(password, salt)
        
        try:
            decrypted_data = Fernet(key).decrypt(encrypted_data)
        except InvalidToken:
            raise ValueError(f"工作流已加密，请输入正确的密码.请前往公众号“阿泰ATAI动态视觉”购买课程获取密码.")
        except Exception as e:
            raise ValueError(f"工作流出错: {str(e)}")
        
        # 写入临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(lora_path)[1]) as tf:
            tf.write(decrypted_data)
            temp_path = tf.name

        try:
            # 使用安全加载加载数据到内存
            lora_data = comfy_utils.load_torch_file(temp_path, safe_load=True)
            
            # 数据成功加载后立即删除临时文件
            try:
                os.unlink(temp_path)
            except Exception as e:
                # 删除失败时添加到全局清理列表作为备用，不记录任何日志
                if temp_path not in temp_files_to_clean:
                    temp_files_to_clean.append(temp_path)
                
            return lora_data
        except Exception as e:
            # 加载失败时尝试删除临时文件并记录日志
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.debug(f"加载失败，已清理临时文件: {temp_path}")
                except:
                    pass
            raise ValueError(f"加载解密后的LoRA文件失败: {str(e)}")

# 仅模型的解密加载器，继承自LoraDecryptLoader
class LoraDecryptLoaderModelOnly(LoraDecryptLoader):
    """仅模型的解密加载器，继承自LoraDecryptLoader"""
    @classmethod
    def INPUT_TYPES(s):
        lora_files = get_all_lora_files(include_none=True)
        return {
            "required": { 
                "model": ("MODEL",),
                "lora_name": (lora_files, {"tooltip": "The name of the LoRA. Select 'None' to not load any LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "password": ("STRING", {"default": "", "placeholder": "加密文件密码", "password": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only"

    def load_lora_model_only(self, model, lora_name, strength_model, password=""):
        if lora_name == "None":
            return (model,)
            
        # 临时抑制ComfyUI的LoRA警告日志（仅抑制特定的未加载键警告）
        lora_logger = logging.getLogger()  # 获取根日志记录器
        
        # 创建一个过滤器来只抑制特定的警告
        class LoraWarningFilter(logging.Filter):
            def filter(self, record):
                # 只过滤掉LoRA未加载键的警告
                if record.levelno == logging.WARNING and "lora key not loaded" in record.getMessage():
                    return False  # 过滤掉这条日志
                return True  # 允许其他日志通过
        
        # 添加过滤器
        warning_filter = LoraWarningFilter()
        for handler in lora_logger.handlers:
            handler.addFilter(warning_filter)
        
        try:
            # 通过预处理增强兼容性
            loaded_model, _ = self.load_lora(model, None, lora_name, strength_model, 0, password)
        finally:
            # 移除过滤器
            for handler in lora_logger.handlers:
                handler.removeFilter(warning_filter)
            
        return (loaded_model,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "LoraEncryptor": LoraEncryptor,
    "LoraBatchEncryptor": LoraBatchEncryptor,
    "LoraDecryptLoader": LoraDecryptLoader,
    "LoraDecryptLoaderModelOnly": LoraDecryptLoaderModelOnly
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraEncryptor": "LoRA处理器",
    "LoraBatchEncryptor": "LoRA批量处理器",
    "LoraDecryptLoader": "LoRA加载器",
    "LoraDecryptLoaderModelOnly": "LoRA加载器（仅模型）"
}

logger.info("The Vertin_tools node has been loaded completely.")
    

import os
import torch
import io
import logging
import tempfile
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import folder_paths
import comfy.utils
import comfy.sd

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("UniversalLoraLoader")

# 加密文件标记（统一使用一个标记确保兼容性）
ENCRYPTION_MARKER = b"LORA_ENC_V1"

# 处理safetensors依赖
try:
    from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    logger.warning("未安装safetensors，无法处理.safetensors格式文件")

# 密钥生成函数
def generate_key(password: str, salt: bytes = None) -> tuple[bytes, bytes]:
    if salt is None:
        salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt

# 检查文件是否为加密文件
def is_encrypted_file(file_path: str) -> bool:
    if not os.path.isfile(file_path) or os.path.getsize(file_path) < len(ENCRYPTION_MARKER) + 32:
        return False
        
    try:
        with open(file_path, "rb") as f:
            return f.read(len(ENCRYPTION_MARKER)) == ENCRYPTION_MARKER
    except:
        return False

# 递归获取所有LoRA文件
def get_all_lora_files(include_none=False):
    lora_files = []
    valid_extensions = ('.safetensors', '.ckpt', '.bin', '.pth', '.pt', '.lora')
    lora_dirs = folder_paths.get_folder_paths("loras") or []
    
    # 添加None选项
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

# 获取文件的绝对路径
def get_lora_absolute_path(relative_path):
    # 处理None选项
    if relative_path == "None":
        return None
        
    lora_dirs = folder_paths.get_folder_paths("loras") or []
    
    if not lora_dirs:
        default_lora_dir = os.path.join(folder_paths.models_dir, "loras")
        if os.path.isdir(default_lora_dir):
            lora_dirs = [default_lora_dir]
    
    for root_dir in lora_dirs:
        absolute_path = os.path.join(root_dir, relative_path)
        if os.path.exists(absolute_path) and os.path.isfile(absolute_path):
            return absolute_path
    
    return None

# 获取LoRA主目录
def get_lora_main_directory():
    lora_dirs = folder_paths.get_folder_paths("loras") or []
    
    if lora_dirs:
        return lora_dirs[0]
    
    default_lora_dir = os.path.join(folder_paths.models_dir, "loras")
    os.makedirs(default_lora_dir, exist_ok=True)
    return default_lora_dir

# 改进的LoRA应用函数，确保权重正确应用
def apply_lora_to_model(model, lora_data, strength):
    """增强版LoRA应用函数，解决未生效问题"""
    # 确保获取原始模型
    if hasattr(model, 'model'):  # 处理ModelPatcher
        original_model = model.model
        logger.debug("检测到ModelPatcher，使用内部model对象")
    else:
        original_model = model
        logger.debug("使用原始模型对象")
    
    # 获取模型状态字典
    model_sd = original_model.state_dict()
    model_keys = set(model_sd.keys())
    logger.debug(f"模型包含 {len(model_keys)} 个参数")
    
    # 收集LoRA参数对
    lora_pairs = {}
    for name in lora_data.keys():
        if 'lora_down' in name:
            base_name = name.replace('.lora_down', '')
            up_name = name.replace('lora_down', 'lora_up')
            if up_name in lora_data:
                lora_pairs[base_name] = (name, up_name)
    
    logger.debug(f"发现 {len(lora_pairs)} 对LoRA参数")
    
    # 跟踪匹配到的参数
    matched_count = 0
    applied_count = 0
    
    # 尝试匹配并应用LoRA
    for base_name, (down_name, up_name) in lora_pairs.items():
        # 尝试多种匹配策略
        possible_matches = [
            base_name,  # 精确匹配
            base_name.replace("transformer.", ""),  # 移除transformer前缀
            f"model.{base_name}",  # 添加model前缀
            base_name.replace(".", "_")  # 替换点为下划线
        ]
        
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
        model_param = model_sd[matched_key]
        
        try:
            # 关键修复：正确计算LoRA权重
            with torch.no_grad():
                # 确保矩阵形状兼容
                if len(lora_down.shape) == 2 and len(lora_up.shape) == 2:
                    # 标准LoRA矩阵形状 (out_features, rank) 和 (rank, in_features)
                    if lora_up.shape[1] == lora_down.shape[0]:
                        lora_weight = torch.matmul(lora_up, lora_down) * strength
                    # 有些LoRA可能是 (rank, out_features) 和 (in_features, rank)
                    elif lora_down.shape[1] == lora_up.shape[0]:
                        lora_weight = torch.matmul(lora_down, lora_up) * strength
                    else:
                        logger.warning(f"LoRA矩阵形状不兼容: {lora_down.shape} 和 {lora_up.shape}")
                        continue
                else:
                    logger.warning(f"不支持的LoRA参数形状: {lora_down.shape}")
                    continue
                
                # 确保权重形状与模型参数匹配
                if lora_weight.shape == model_param.shape:
                    model_sd[matched_key] += lora_weight
                    applied_count += 1
                    logger.debug(f"成功应用LoRA到 {matched_key} (强度: {strength})")
                else:
                    # 尝试调整形状
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
        
        # 特殊处理：如果是ModelPatcher，触发更新
        if hasattr(model, 'model'):
            model.model = original_model
            logger.debug("已更新ModelPatcher中的模型")
            
        return model
    except Exception as e:
        logger.error(f"加载修改后的状态字典时出错: {str(e)}")
        return model

# 单个文件加密函数 - 确保加密文件保存在原文件相同目录
def encrypt_single_file(lora_path, password, output_suffix, overwrite, skip_validation):
    lora_filename = os.path.basename(lora_path)
    file_dir = os.path.dirname(lora_path)  # 获取原文件所在目录
    file_ext = os.path.splitext(lora_path)[1].lower()
    
    if is_encrypted_file(lora_path):
        return f"跳过: 已加密 {lora_filename}"
    
    valid_extensions = ('.safetensors', '.ckpt', '.bin', '.pth', '.pt', '.lora')
    if not skip_validation and not (file_ext in valid_extensions and os.path.isfile(lora_path)):
        return f"跳过: 无效文件 {lora_filename}"
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file_path = temp_file.name
        
        with open(lora_path, 'rb') as f:
            original_content = f.read()
        
        if file_ext == '.safetensors' and SAFETENSORS_AVAILABLE:
            try:
                lora_data = load_safetensors(lora_path, device='cpu')
                save_safetensors(lora_data, temp_file_path)
            except Exception as e:
                if skip_validation:
                    logger.warning(f"文件 {lora_filename} 验证警告: {str(e)}，但已选择跳过验证")
                    with open(temp_file_path, 'wb') as f:
                        f.write(original_content)
                else:
                    os.unlink(temp_file_path)
                    raise ValueError(f"无法加载safetensors文件: {str(e)}")
        else:
            try:
                lora_data = torch.load(lora_path, map_location="cpu", weights_only=False)
                torch.save(lora_data, temp_file_path)
            except Exception as e:
                if skip_validation:
                    logger.warning(f"文件 {lora_filename} 验证警告: {str(e)}，但已选择跳过验证")
                    with open(temp_file_path, 'wb') as f:
                        f.write(original_content)
                else:
                    os.unlink(temp_file_path)
                    raise ValueError(f"无法加载文件: {str(e)}")
        
        with open(temp_file_path, 'rb') as f:
            content_to_encrypt = f.read()
        
        key, salt = generate_key(password)
        encrypted_data = Fernet(key).encrypt(content_to_encrypt)
        
        # 构建输出文件名，确保保存在原目录，默认后缀为enc
        base_name = os.path.splitext(lora_filename)[0]
        clean_suffix = output_suffix.strip('_')
        output_name = f"{base_name}_{clean_suffix}{file_ext}" if clean_suffix else f"{base_name}_enc{file_ext}"
        output_path = os.path.join(file_dir, output_name)  # 明确使用原文件所在目录
        
        if os.path.exists(output_path) and not overwrite:
            os.unlink(temp_file_path)
            return f"跳过: 已存在 {output_name}（保存在原文件目录）"
        
        with open(output_path, "wb") as f:
            f.write(ENCRYPTION_MARKER + b"|||" + salt + b"|||" + encrypted_data)
        
        os.unlink(temp_file_path)
        
        logger.info(f"加密成功: {lora_filename} → {output_name}（保存在原文件目录）")
        return f"成功: {lora_filename} → {output_name}（保存在原文件目录）"
        
    except Exception as e:
        error_msg = f"失败: {lora_filename} - {str(e)}"
        logger.error(error_msg)
        return error_msg

# LoRA加密器节点
class LoraEncryptor:
    def __init__(self):
        self.lora_dirs = folder_paths.get_folder_paths("loras") or []

    @classmethod
    def INPUT_TYPES(s):
        # 加密器不需要None选项，所以传入include_none=False
        all_lora_files = get_all_lora_files(include_none=False)
        return {
            "required": {
                "lora文件名": (all_lora_files,),
                "加密密码": ("STRING", {
                    "default": "", "placeholder": "加密密码", "password": True
                }),
                "输出文件后缀": ("STRING", {
                    "default": "enc", "placeholder": "例如：enc"  # 默认后缀为enc
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
            
        # 调用加密函数，加密文件将自动保存到原文件所在目录
        return (encrypt_single_file(lora_path, 加密密码, 输出文件后缀, 覆盖已存在文件, 跳过验证),)

# LoRA批量加密器节点（移除常用目录选择）
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
                    "default": "enc", "placeholder": "例如：enc"  # 默认后缀为enc
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
        # 使用用户输入的目录作为目标目录
        target_dir = 加密目录.strip()
            
        if not target_dir or not os.path.isdir(target_dir):
            return ("错误: 请输入有效的文件夹路径",)
        if not 加密密码:
            return ("错误: 请输入加密密码",)
            
        # 保存最近使用的目录
        if target_dir not in self.last_used_dirs:
            self.last_used_dirs.insert(0, target_dir)
            # 限制最近使用目录的数量
            if len(self.last_used_dirs) > 10:
                self.last_used_dirs = self.last_used_dirs[:10]
            
        results = []
        valid_extensions = ('.safetensors', '.ckpt', '.bin', '.pth', '.pt', '.lora')
        
        # 递归扫描目录并加密 - 每个文件的加密结果将保存在其原目录
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
                        # 检查是否是有效模型文件且未加密
                        if (file_ext in valid_extensions and 
                            not is_encrypted_file(item_path)):
                            # 调用单个加密函数，自动将加密文件保存到原文件所在目录
                            result = encrypt_single_file(
                                item_path, 加密密码, 输出文件后缀, 
                                覆盖已存在文件, 跳过验证
                            )
                            results.append(result)
            except PermissionError:
                results.append(f"警告: 没有权限访问目录 {directory}")
            except Exception as e:
                results.append(f"错误: 扫描目录 {directory} 时出错 - {str(e)}")
        
        # 开始扫描和加密
        scan_and_encrypt(target_dir)
        
        # 汇总结果
        total = len(results)
        success = sum(1 for r in results if r.startswith("成功:"))
        skipped = sum(1 for r in results if r.startswith("跳过:"))
        failed = sum(1 for r in results if r.startswith("失败:"))
        
        summary = (f"批量加密完成 - 总计: {total}, 成功: {success}, "
                  f"跳过: {skipped}, 失败: {failed}\n所有加密文件均保存在原文件所在目录\n\n详细结果:\n")
        summary += "\n".join(results)
        
        return (summary,)

# 带解密功能的LoRA加载器
class LoraDecryptLoader:
    """带解密功能的LoRA加载器，保留原生LoraLoader全部逻辑"""
    def __init__(self):
        self.loaded_lora = None  # 缓存 (路径, 数据, 密码)

    @classmethod
    def INPUT_TYPES(s):
        # 在加载器中添加None选项
        lora_files = get_all_lora_files(include_none=True)
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (lora_files, {"tooltip": "The name of the LoRA. Select 'None' to not load any LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. Range 0.0-1.0."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. Range 0.0-1.0."}),
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
        # 如果选择了None，直接返回原始模型
        if lora_name == "None":
            return (model, clip)
            
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        
        # 检查缓存（包含密码，确保加密文件缓存正确）
        if self.loaded_lora is not None:
            cached_path, cached_lora, cached_password = self.loaded_lora
            if cached_path == lora_path and cached_password == password:
                lora = cached_lora
            else:
                self.loaded_lora = None

        if lora is None:
            # 加载LoRA数据（自动处理加密/未加密）
            lora = self._load_lora_data(lora_path, password)
            self.loaded_lora = (lora_path, lora, password)

        # 使用原生方法应用LoRA（保持原生逻辑不变）
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

    def _load_lora_data(self, lora_path: str, password: str):
        """加载LoRA数据，自动检测加密状态"""
        if is_encrypted_file(lora_path):
            # 加密文件处理
            if not password:
                raise ValueError(f"文件 {os.path.basename(lora_path)} 已加密，请提供密码")
            return self._decrypt_lora(lora_path, password)
        else:
            # 未加密文件：完全使用原生加载逻辑
            return comfy.utils.load_torch_file(lora_path, safe_load=True)

    def _decrypt_lora(self, lora_path: str, password: str):
        """解密LoRA文件并返回数据，错误密码时明确提示"""
        with open(lora_path, "rb") as f:
            data = f.read()

        # 解析加密结构
        parts = data.split(b"|||", 2)
        if len(parts) != 3 or parts[0] != ENCRYPTION_MARKER:
            raise ValueError("无效的加密LoRA文件格式")
        
        _, salt, encrypted_data = parts
        
        # 生成密钥
        key, _ = generate_key(password, salt)
        
        try:
            # 解密数据（捕获密码错误）
            decrypted_data = Fernet(key).decrypt(encrypted_data)
        except InvalidToken:
            # 明确提示密码错误
            raise ValueError(f"解密失败：文件 {os.path.basename(lora_path)} 的密码不正确，请检查密码")
        except Exception as e:
            raise ValueError(f"解密过程出错: {str(e)}")
        
        # 写入临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(lora_path)[1]) as tf:
            tf.write(decrypted_data)
            temp_path = tf.name

        try:
            # 使用原生方法加载解密后的文件
            return comfy.utils.load_torch_file(temp_path, safe_load=True)
        finally:
            os.unlink(temp_path)  # 清理临时文件


class LoraDecryptLoaderModelOnly(LoraDecryptLoader):
    """仅模型的解密加载器，继承自LoraDecryptLoader"""
    @classmethod
    def INPUT_TYPES(s):
        # 在仅模型的加载器中也添加None选项
        lora_files = get_all_lora_files(include_none=True)
        return {
            "required": { 
                "model": ("MODEL",),
                "lora_name": (lora_files, {"tooltip": "The name of the LoRA. Select 'None' to not load any LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "password": ("STRING", {"default": "", "placeholder": "加密文件密码", "password": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only"

    def load_lora_model_only(self, model, lora_name, strength_model, password=""):
        # 如果选择了None，直接返回原始模型
        if lora_name == "None":
            return (model,)
            
        # 调用父类方法，仅返回模型结果
        return (self.load_lora(model, None, lora_name, strength_model, 0, password)[0],)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "LoraEncryptor": LoraEncryptor,
    "LoraBatchEncryptor": LoraBatchEncryptor,
    "LoraDecryptLoader": LoraDecryptLoader,
    "LoraDecryptLoaderModelOnly": LoraDecryptLoaderModelOnly
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraEncryptor": "LoRA加密器",
    "LoraBatchEncryptor": "LoRA批量加密器",
    "LoraDecryptLoader": "LoRA解密加载器",
    "LoraDecryptLoaderModelOnly": "LoRA解密加载器（仅模型）"
}

logger.info("LoRA工具节点加载完成")

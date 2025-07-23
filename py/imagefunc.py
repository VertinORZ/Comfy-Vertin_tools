import os
import random
import string

def log(message, message_type='info'):
    """简单的日志函数"""
    print(f"[{message_type.upper()}] {message}")

def generate_random_name(prefix="", suffix="", length=16):
    """生成随机字符串作为临时文件名"""
    chars = string.ascii_letters + string.digits
    random_str = ''.join(random.choice(chars) for _ in range(length))
    return f"{prefix}{random_str}{suffix}"

def remove_empty_lines(text):
    """移除文本中的空行"""
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join([line for line in lines if line])
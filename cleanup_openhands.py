#!/usr/bin/env python3
"""
OpenHands 清理和修复脚本
删除不需要的大文件，安装缺失的依赖
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def install_missing_dependencies():
    """安装缺失的依赖包"""
    print("🔧 安装缺失的依赖包...")
    
    missing_packages = [
        "litellm",
        "toml"
    ]
    
    for package in missing_packages:
        try:
            print(f"正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 安装失败: {e}")
            return False
    
    return True

def cleanup_large_directories():
    """清理大目录"""
    print("\n🗑️  清理大目录...")
    
    openhands_dir = Path("cellforge/Code_Generation/OpenHands")
    
    # 可以安全删除的目录
    deletable_dirs = [
        ".git",           # Git历史 - 159.4 MB
        "docs",           # 文档 - 29.9 MB  
        "evaluation",     # 评估文件 - 6.8 MB
        "frontend",       # 前端文件 - 2.2 MB
        "tests",          # 测试文件 - 1.3 MB
        ".github",        # GitHub配置 - 0.1 MB
        "containers",     # 容器文件 - 0.0 MB
        "dev_config",     # 开发配置 - 0.0 MB
        "microagents"     # 微代理 - 0.0 MB
    ]
    
    total_saved = 0
    
    for dir_name in deletable_dirs:
        dir_path = openhands_dir / dir_name
        if dir_path.exists():
            try:
                # 计算目录大小
                size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                
                # 删除目录
                shutil.rmtree(dir_path)
                print(f"✅ 删除 {dir_name}/ - 节省 {size_mb:.1f} MB")
                total_saved += size_mb
                
            except Exception as e:
                print(f"❌ 删除 {dir_name}/ 失败: {e}")
    
    print(f"\n💾 总共节省空间: {total_saved:.1f} MB")
    return True

def cleanup_large_files():
    """清理大文件"""
    print("\n🗑️  清理大文件...")
    
    openhands_dir = Path("cellforge/Code_Generation/OpenHands")
    
    # 可以删除的大文件
    deletable_files = [
        "poetry.lock",    # 锁定文件，可以重新生成
        "pydoc-markdown.yml",
        "pytest.ini",
        "build.sh",
        "Makefile",
        "ISSUE_TRIAGE.md",
        "Development.md",
        "COMMUNITY.md", 
        "CONTRIBUTING.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        ".nvmrc",
        ".dockerignore",
        ".gitattributes"
    ]
    
    for file_name in deletable_files:
        file_path = openhands_dir / file_name
        if file_path.exists():
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                file_path.unlink()
                print(f"✅ 删除 {file_name} - 节省 {size_mb:.1f} MB")
            except Exception as e:
                print(f"❌ 删除 {file_name} 失败: {e}")

def create_minimal_config():
    """创建最小化的配置文件"""
    print("\n⚙️  创建最小化配置...")
    
    config_content = """# OpenHands 最小化配置
[core]
workspace_base = "./workspace"
cache_dir = "/tmp/cache"
debug = false
max_iterations = 100
runtime = "docker"
default_agent = "CodeActAgent"

[llm]
api_key = ""
base_url = ""
model = "gpt-4o"
temperature = 0.1
max_message_chars = 50000
max_input_tokens = 0
max_output_tokens = 0
num_retries = 3
retry_max_wait = 60
retry_min_wait = 10
retry_multiplier = 2.0
drop_params = false
modify_params = true
caching_prompt = true
top_p = 1.0
disable_vision = true

[agent]
codeact_enable_browsing = true
codeact_enable_llm_editor = false
codeact_enable_jupyter = true
enable_prompt_extensions = true
"""
    
    config_path = Path("cellforge/Code_Generation/OpenHands/config.toml")
    try:
        with open(config_path, 'w') as f:
            f.write(config_content)
        print("✅ 配置文件已更新")
    except Exception as e:
        print(f"❌ 配置文件更新失败: {e}")

def test_openhands_startup():
    """测试OpenHands启动"""
    print("\n🧪 测试OpenHands启动...")
    
    openhands_dir = Path("cellforge/Code_Generation/OpenHands")
    
    try:
        # 检查基本模块导入
        sys.path.insert(0, str(openhands_dir))
        
        # 测试基本导入
        import openhands
        print("✅ openhands 模块导入成功")
        
        # 测试配置文件
        import toml
        with open(openhands_dir / "config.toml", 'r') as f:
            config = toml.load(f)
        print("✅ 配置文件解析成功")
        
        # 检查可执行文件
        if (openhands_dir / "openhands").exists():
            print("✅ OpenHands 可执行文件存在")
        else:
            print("⚠️  OpenHands 可执行文件不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 启动测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 OpenHands 清理和修复")
    print("=" * 50)
    
    # 检查当前目录
    if not Path("cellforge/Code_Generation/OpenHands").exists():
        print("❌ 未找到OpenHands目录")
        return
    
    # 询问用户是否继续
    print("\n⚠️  此操作将删除以下内容:")
    print("  - .git/ (Git历史)")
    print("  - docs/ (文档)")
    print("  - evaluation/ (评估文件)")
    print("  - frontend/ (前端文件)")
    print("  - tests/ (测试文件)")
    print("  - 其他开发相关文件")
    print("\n预计可节省约 200MB 空间")
    
    response = input("\n是否继续? (y/N): ").strip().lower()
    if response != 'y':
        print("操作已取消")
        return
    
    # 执行清理操作
    try:
        # 1. 安装缺失的依赖
        if not install_missing_dependencies():
            print("❌ 依赖安装失败")
            return
        
        # 2. 清理大目录
        cleanup_large_directories()
        
        # 3. 清理大文件
        cleanup_large_files()
        
        # 4. 创建最小化配置
        create_minimal_config()
        
        # 5. 测试启动
        if test_openhands_startup():
            print("\n🎉 清理完成！OpenHands现在应该可以正常启动了")
            print("\n💡 使用建议:")
            print("  1. 运行: cd cellforge/Code_Generation/OpenHands && ./openhands")
            print("  2. 或者使用Docker: docker run -it openhands/openhands")
            print("  3. 如果需要文档，可以从GitHub重新克隆")
        else:
            print("\n⚠️  清理完成，但启动测试失败，可能需要进一步配置")
            
    except Exception as e:
        print(f"❌ 清理过程中出现错误: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time

if "pythonw" in sys.executable.lower():
    sys.stdout = open(os.devnull, "w", encoding='utf-8')
    sys.stderr = open(os.devnull, "w", encoding='utf-8')

from loguru import logger

logger.remove()
from .i18n import set_lang, T
from .aipy import CONFIG_DIR, ConfigManager
logger.add(CONFIG_DIR / "aipyapp.log", format="{time:HH:mm:ss} | {level} | {message} | {extra}", level='INFO')

def parse_args():
    import argparse
    config_help_message = (
        f"Specify the configuration directory.\nDefaults to {CONFIG_DIR} if not provided."
    )

    # 创建共享的参数组
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument('--style', default=None, help="Style of the display, e.g. 'classic' or 'modern'")
    common_args.add_argument('--role', default=None, help="Role to use")

    parser = argparse.ArgumentParser(description="Python use - AIPython", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-c", '--config-dir', default=CONFIG_DIR, type=str, help=config_help_message)
    parser.add_argument('--debug', default=False, action='store_true', help="Debug mode")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # update 子命令 - 不需要 style/role
    update_parser = subparsers.add_parser('update', help='Update aipyapp to latest version')
    update_parser.add_argument('--beta', action='store_true', help='Include beta versions in update')

    # sync 子命令 - 不需要 style/role
    sync_parser = subparsers.add_parser('sync', help='Sync content from trustoken')

    # python 子命令 - 继承 common_args
    python_parser = subparsers.add_parser('python', help='Python mode', parents=[common_args])

    # ipython 子命令 - 继承 common_args
    ipython_parser = subparsers.add_parser('ipython', help='IPython mode', parents=[common_args])

    # gui 子命令 - 继承 common_args
    gui_parser = subparsers.add_parser('gui', help='GUI mode', parents=[common_args])

    # run 子命令 - 继承 common_args，合并原来的 exec 和 run 功能
    run_parser = subparsers.add_parser('run', help='CMD mode - execute instruction or run JSON file', parents=[common_args])
    run_parser.add_argument('instruction', nargs='?', help='Instruction to execute')
    run_parser.add_argument('--task', default=None, help='JSON file to run as task context')

    # agent 子命令 - 只需要 port/host
    agent_parser = subparsers.add_parser('agent', help='Agent mode - HTTP API server for n8n integration')
    agent_parser.add_argument('--port', type=int, default=8848, help="Port for agent mode HTTP server (default: 8848)")
    agent_parser.add_argument('--host', default='127.0.0.1', help="Host for agent mode HTTP server (default: 127.0.0.1)")
    
    return parser.parse_args()

def ensure_pkg(pkg):
    try:
        if pkg == 'wxpython':
            import wx
        elif pkg == 'ipython':
            import IPython
        elif pkg == 'fastapi':
            import fastapi
        elif pkg == 'uvicorn':
            import uvicorn
    except ImportError:
        import subprocess
        print(f"Installing required package: {pkg}")
        cp = subprocess.run([sys.executable, "-m", "pip", "install", pkg])
        assert cp.returncode == 0

def handle_update(args):
    """处理 update 命令"""
    import subprocess
    from . import __version__
    
    package_name = 'aipyapp'
    print(f"当前版本: {__version__}")
    
    if args.beta:
        print(f"更新到最新版本 (包括测试版): {package_name}")
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--pre", package_name]
    else:
        print(f"更新到最新稳定版本: {package_name}")
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package_name]
    
    # China mirror
    if time.timezone / 3600 == -8:
        os.environ['PIP_INDEX_URL'] = 'https://mirrors.aliyun.com/pypi/simple'
        os.environ['PIP_EXTRA_INDEX_URL'] = 'https://pypi.tuna.tsinghua.edu.cn/simple'

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("更新完成!")
        if result.stdout.strip():
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"更新失败: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"更新失败: {str(e)}")
        sys.exit(1)

def handle_sync(conf, args):
    """处理 sync 命令"""
    conf.fetch_config()

def init_settings(conf, args):
    settings = conf.get_config()
    lang = settings.get('lang')
    if lang: set_lang(lang)
    
    # 根据子命令设置gui模式
    command = getattr(args, 'command', None)
    settings.gui = (command == 'gui')
    settings.debug = args.debug
    settings.config_dir = args.config_dir
    
    # 处理 role 和 style 参数（只有部分子命令支持）
    if hasattr(args, 'role') and args.role:
        settings['role'] = args.role.lower()
    if hasattr(args, 'style') and args.style:
        display_config = settings.setdefault('display', {})
        display_config['style'] = args.style
    
    # 处理 agent 模式的特殊参数
    if command == 'agent':
        settings['agent'] = {'port': args.port, 'host': args.host}

    #TODO: remove these lines
    if conf.check_config(gui=True) == 'TrustToken':
        from .config import LLMConfig
        llm_config = LLMConfig(CONFIG_DIR / "config")
        if llm_config.need_config():
            settings['llm_need_config'] = True
            if not args.gui:
                from .aipy.wizard import config_llm
                config_llm(llm_config)
                if llm_config.need_config():
                    print(f"❌ {T('LLM configuration required')}")
                    sys.exit(1)
        settings["llm"] = llm_config.config
        
    settings['config_manager'] = conf
    return settings

def get_aipy_main(args, settings):
    """根据参数获取对应的 aipy_main 函数"""
    command = getattr(args, 'command', None)
    
    if command == 'agent':
        ensure_pkg('fastapi')
        ensure_pkg('uvicorn')
        from .cli.cli_agent import main as aipy_main
    elif command == 'python':
        from .cli.cli_python import main as aipy_main
    elif command == 'ipython':
        ensure_pkg('ipython')
        from .cli.cli_ipython import main as aipy_main
    elif command == 'gui':
        settings['gui'] = True
        ensure_pkg('wxpython')
        from .gui.main import main as aipy_main
    elif command == 'run':
        if args.instruction:
            settings['exec_cmd'] = args.instruction
        if args.task:
            settings['run_json'] = args.task
        from .cli.cli_task import main as aipy_main
    else:
        # 默认进入 task 模式
        from .cli.cli_task import main as aipy_main
    return aipy_main

def main():
    args = parse_args()
    
    # 处理特殊子命令
    if args.command == 'update':
        handle_update(args)
        return
    elif args.command == 'sync':
        conf = ConfigManager(args.config_dir)
        handle_sync(conf, args)
        return
    
    # 验证 run 命令参数
    if args.command == 'run':
        if not args.instruction and not args.task:
            print("Error: run command requires either an instruction or --task option")
            return 1
    
    # 处理其他命令
    conf = ConfigManager(args.config_dir)
    settings = init_settings(conf, args)
    aipy_main = get_aipy_main(args, settings)
    aipy_main(settings)

def mainw():
    args = parse_args()
    ensure_pkg('wxpython')
    from .gui.main import main as aipy_main
    aipy_main(args)

if __name__ == '__main__':
    main()

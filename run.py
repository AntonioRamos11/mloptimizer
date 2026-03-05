#!/usr/bin/env python3
"""
MLOptimizer - Unified Runner
=============================
Single script to:
1. Validate and install requirements
2. Run master/slave nodes
3. Support local and cloud deployment

Usage:
    python run.py                    # Full run (master + slave)
    python run.py --master          # Master only
    python run.py --slave            # Slave only
    python run.py --check            # Just validate requirements
    python run.py --install          # Install requirements only
    python run.py --help             # Show all options
"""

import subprocess
import sys
import os
import argparse
import time
import signal
from pathlib import Path
from typing import List, Tuple, Optional

REQUIREMENTS_FILES = ["requirements.txt", "requirements2.txt"]
VENV_DIR = "venv_mlopt"


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_status(msg: str, status: str = "info"):
    symbols = {"info": "ℹ", "ok": "✓", "error": "✗", "warn": "⚠", "step": "→"}
    colors = {
        "info": Colors.BLUE,
        "ok": Colors.GREEN,
        "error": Colors.RED,
        "warn": Colors.YELLOW,
        "step": Colors.BOLD + Colors.BLUE
    }
    color = colors.get(status, "")
    symbol = symbols.get(status, "•")
    print(f"{color}{symbol} {msg}{Colors.END}")


def run_command(cmd: List[str], cwd: str = None, capture: bool = False, env: dict = None) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture,
            text=True,
            env=merged_env
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError as e:
        return 1, "", str(e)
    except Exception as e:
        return 1, "", str(e)


def get_python_cmd() -> str:
    """Find the best available Python command."""
    python_cmds = ["python3.12", "python3.11", "python3.10", "python3", "python"]
    
    for cmd in python_cmds:
        code, _, _ = run_command([cmd, "--version"], capture=True)
        if code == 0:
            version_code, stdout, _ = run_command([cmd, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], capture=True)
            if version_code == 0:
                version = stdout.strip()
                if tuple(map(int, version.split('.'))) >= (3, 10):
                    print_status(f"Found Python {version}: {cmd}", "ok")
                    return cmd
    
    print_status("Python 3.10+ not found!", "error")
    return "python3"


def get_venv_python() -> Optional[str]:
    """Get Python from virtual environment if it exists."""
    venv_python = Path(VENV_DIR) / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return None


def create_venv(python_cmd: str) -> bool:
    """Create virtual environment."""
    print_status(f"Creating virtual environment: {VENV_DIR}", "step")
    
    code, stdout, stderr = run_command([python_cmd, "-m", "venv", VENV_DIR], capture=True)
    
    if code == 0:
        print_status(f"Virtual environment created", "ok")
        return True
    else:
        print_status(f"Failed to create venv: {stderr}", "error")
        
        # Try alternative method
        print_status("Trying alternative method...", "warn")
        code, stdout, stderr = run_command([
            python_cmd, "-m", "venv", "--without-pip", VENV_DIR
        ], capture=True)
        
        if code == 0:
            # Install pip manually
            get_pip = subprocess.run(
                "curl -sS https://bootstrap.pypa.io/get-pip.py | " + python_cmd,
                shell=True, capture_output=True, text=True
            )
            if get_pip.returncode == 0:
                print_status("Pip installed manually", "ok")
                return True
        
        return False


def get_installed_packages() -> set:
    """Get set of currently installed packages."""
    packages = set()
    code, stdout, _ = run_command([sys.executable, "-m", "pip", "list", "--format=freeze"], capture=True)
    
    if code == 0:
        for line in stdout.splitlines():
            if "==" in line:
                pkg = line.split("==")[0].lower().replace("-", "_")
                packages.add(pkg)
    
    return packages


def read_requirements() -> List[Tuple[str, str]]:
    """Read all requirements from both files."""
    all_reqs = []
    
    for req_file in REQUIREMENTS_FILES:
        if Path(req_file).exists():
            print_status(f"Reading {req_file}", "info")
            with open(req_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Parse package name and version
                        pkg = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
                        all_reqs.append((pkg, line))
    
    # Remove duplicates
    seen = set()
    unique_reqs = []
    for pkg, full_line in all_reqs:
        if pkg.lower() not in seen:
            seen.add(pkg.lower())
            unique_reqs.append((pkg, full_line))
    
    return unique_reqs


def check_requirements(installed: set) -> Tuple[List[str], List[str]]:
    """Check which requirements are missing."""
    all_reqs = read_requirements()
    
    missing = []
    present = []
    
    for pkg, full_line in all_reqs:
        pkg_normalized = pkg.lower().replace("-", "_")
        if pkg_normalized not in installed:
            missing.append(full_line)
        else:
            present.append(pkg)
    
    return missing, present


def install_requirements(use_venv: bool = True) -> bool:
    """Install all requirements."""
    print_status("=" * 50)
    print_status("Installing Requirements")
    print_status("=" * 50)
    
    # Determine which Python to use
    if use_venv:
        python_cmd = get_venv_python()
        if python_cmd is None:
            base_python = get_python_cmd()
            if not create_venv(base_python):
                print_status("Failed to create venv, using system Python", "warn")
                python_cmd = base_python
            else:
                python_cmd = get_venv_python()
    else:
        python_cmd = sys.executable
    
    if python_cmd is None:
        python_cmd = "python3"
    
    print_status(f"Using Python: {python_cmd}", "info")
    
    # Upgrade pip first
    print_status("Upgrading pip...", "step")
    run_command([python_cmd, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], capture=False)
    
    # Check what's already installed
    print_status("Checking installed packages...", "step")
    code, stdout, _ = run_command([python_cmd, "-m", "pip", "list", "--format=freeze"], capture=True)
    installed = set()
    if code == 0:
        for line in stdout.splitlines():
            if "==" in line:
                pkg = line.split("==")[0].lower().replace("-", "_")
                installed.add(pkg)
    
    # Find missing packages
    all_reqs = read_requirements()
    missing = []
    for pkg, full_line in all_reqs:
        pkg_normalized = pkg.lower().replace("-", "_")
        if pkg_normalized not in installed:
            missing.append(full_line)
    
    if not missing:
        print_status("All requirements already installed!", "ok")
        return True
    
    print_status(f"Need to install {len(missing)} packages", "info")
    
    # Install in batches to avoid too long command lines
    batch_size = 50
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i+batch_size]
        print_status(f"Installing batch {i//batch_size + 1}/{(len(missing)-1)//batch_size + 1} ({len(batch)} packages)...", "step")
        
        cmd = [python_cmd, "-m", "pip", "install", "--upgrade"] + batch
        code, stdout, stderr = run_command(cmd, capture=True)
        
        if code != 0:
            print_status(f"Error installing batch: {stderr[:200]}", "error")
    
    # Verify TensorFlow
    print_status("Verifying TensorFlow...", "step")
    code, stdout, stderr = run_command([
        python_cmd, "-c", 
        "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); "
        "gpus = tf.config.list_physical_devices('GPU'); "
        "print(f'GPUs: {len(gpus)}')"
    ], capture=True)
    
    if code == 0:
        print_status(f"TensorFlow verified: {stdout.strip()}", "ok")
    else:
        print_status(f"TensorFlow check failed: {stderr[:200]}", "warn")
    
    print_status("Requirements installation complete!", "ok")
    return True


def get_env_vars(args) -> dict:
    """Get environment variables for running the project."""
    env = os.environ.copy()
    
    # Cloud settings
    env["CLOUD_MODE"] = str(args.cloud_mode if hasattr(args, 'cloud_mode') else 1)
    
    # RabbitMQ settings
    if args.host:
        env["INSTANCE_HOST_URL"] = args.host
    if args.port:
        env["INSTANCE_PORT"] = str(args.port)
    if args.mgmt_url:
        env["INSTANCE_MANAGMENT_URL"] = args.mgmt_url
    
    # Dataset
    if args.dataset:
        env["DATASET_NAME"] = args.dataset
    
    # TensorFlow settings
    env["TF_CPP_MIN_LOG_LEVEL"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    
    return env


def run_master(args) -> int:
    """Run the master node."""
    print_status("=" * 50)
    print_status("Starting Master Node")
    print_status("=" * 50)
    
    python_cmd = get_venv_python() or sys.executable
    
    # Get environment
    env = get_env_vars(args)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print_status(f"Python: {python_cmd}", "info")
    print_status(f"Cloud Mode: {env['CLOUD_MODE']}", "info")
    print_status(f"Dataset: {env.get('DATASET_NAME', 'default')}", "info")
    
    # Run master
    print_status("Starting run_master.py...", "step")
    
    return subprocess.call(
        [python_cmd, "-u", "run_master.py"],
        env=env
    )


def run_slave(args) -> int:
    """Run the slave node."""
    print_status("=" * 50)
    print_status("Starting Slave Node")
    print_status("=" * 50)
    
    python_cmd = get_venv_python() or sys.executable
    
    # Get environment
    env = get_env_vars(args)
    gpu = args.gpu if args.gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    print_status(f"Python: {python_cmd}", "info")
    print_status(f"GPU: {gpu}", "info")
    
    # Run slave
    print_status("Starting run_slave.py...", "step")
    
    return subprocess.call(
        [python_cmd, "-u", "run_slave.py"],
        env=env
    )


def run_full(args) -> None:
    """Run both master and slave nodes."""
    print_status("=" * 50)
    print_status("Starting MLOptimizer (Full Mode)")
    print_status("=" * 50)
    
    python_cmd = get_venv_python() or sys.executable
    env = get_env_vars(args)
    
    # Start master
    print_status("Starting Master...", "step")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    master_proc = subprocess.Popen(
        [python_cmd, "-u", "run_master.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    time.sleep(3)
    
    # Start slave
    print_status("Starting Slave...", "step")
    slave_env = env.copy()
    slave_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    slave_proc = subprocess.Popen(
        [python_cmd, "-u", "run_slave.py"],
        env=slave_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    print_status("Both nodes running. Press Ctrl+C to stop.", "ok")
    
    # Handle signals
    def cleanup(signum, frame):
        print_status("Shutting down...", "warn")
        master_proc.terminate()
        slave_proc.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Stream output
    import select
    
    while True:
        reads = [master_proc.stdout, slave_proc.stdout]
        readable, _, _ = select.select(reads, [], [], 1)
        
        for stream in readable:
            line = stream.readline()
            if line:
                print(line, end="")


def main():
    parser = argparse.ArgumentParser(
        description="MLOptimizer - Unified Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --checsk              # Check requirements only
  python run.py --install           # Install requirements only
  python run.py --master            # Run master only
  python run.py --slave             # Run slave only
  python run.py                     # Run full (master + slave)
  
Advanced:
  python run.py --master --host=myhost.com --port=5672
  python run.py --slave --gpu=0 --dataset=cifar10
  python run.py --install --no-venv  # Install to system Python
        """
    )
    
    parser.add_argument("--check", action="store_true",
                        help="Validate requirements only (no install)")
    parser.add_argument("--install", action="store_true",
                        help="Install requirements and exit")
    parser.add_argument("--master", action="store_true",
                        help="Run master node only")
    parser.add_argument("--slave", action="store_true",
                        help="Run slave node only")
    parser.add_argument("--no-venv", action="store_true",
                        help="Use system Python instead of venv")
    
    # Configuration options
    parser.add_argument("--host", type=str, default=None,
                        help="RabbitMQ host URL")
    parser.add_argument("--port", type=int, default=None,
                        help="RabbitMQ port")
    parser.add_argument("--mgmt-url", type=str, default=None,
                        help="RabbitMQ management URL")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (mnist, cifar10, etc)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device ID")
    parser.add_argument("--cloud-mode", type=int, default=1, choices=[0, 1],
                        help="Cloud mode: 0=local, 1=remote (default: 1)")
    
    args = parser.parse_args()
    
    print_status("=" * 50)
    print_status("MLOptimizer - Unified Runner")
    print_status("=" * 50)
    
    # Check only
    if args.check:
        print_status("Checking requirements...", "step")
        installed = get_installed_packages()
        missing, present = check_requirements(installed)
        
        print_status(f"Installed: {len(present)} packages", "ok")
        print_status(f"Missing: {len(missing)} packages", "warn")
        
        if missing:
            print_status("\nMissing packages:", "warn")
            for pkg in missing[:10]:
                print(f"  - {pkg}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
            print_status("\nRun with --install to install missing packages", "info")
        
        return 0
    
    # Install only
    if args.install:
        install_requirements(use_venv=not args.no_venv)
        return 0
    
    # Determine what to run
    run_master_flag = args.master
    run_slave_flag = args.slave
    
    # Default: run both if neither specified
    if not run_master_flag and not run_slave_flag:
        run_master_flag = True
        run_slave_flag = True
    
    # Auto-install if venv doesn't exist or packages missing
    venv_python = get_venv_python()
    if venv_python is None:
        print_status("Virtual environment not found, installing requirements...", "warn")
        install_requirements(use_venv=not args.no_venv)
    
    # Run requested components
    try:
        if run_master_flag and run_slave_flag:
            run_full(args)
        elif run_master_flag:
            return run_master(args)
        elif run_slave_flag:
            return run_slave(args)
    except KeyboardInterrupt:
        print_status("\nInterrupted by user", "warn")
        return 130
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""System- and environment-related helper utilities."""
from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict
import platform
import sys


def get_system_info() -> Dict[str, str]:
    """Collect system information for output log."""
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'working_directory': str(Path.cwd().resolve()),
        'script_path': str(Path(sys.argv[0]).resolve()),
    }

    try:
        result = subprocess.run(['xtb', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'xtb' in line.lower() and any(c.isdigit() for c in line):
                    info['xtb_version'] = line.strip()
                    break
            else:
                info['xtb_version'] = 'Available (version unknown)'
        else:
            info['xtb_version'] = 'Not available or error'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        info['xtb_version'] = 'Not found in PATH'

    return info


def format_file_info(filepath: str) -> str:
    """Format file information including size and modification time."""
    try:
        path = Path(filepath)
        if path.exists():
            stat = path.stat()
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            return f"{path.name} ({size_kb:.1f} KB, modified: {mtime})"
        return f"{path.name} (file not found)"
    except Exception:
        return f"{filepath} (error reading file info)"


def set_thread_env_vars(threads: int) -> None:
    """Set environment variables to control threading in libraries."""
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[var] = str(threads)

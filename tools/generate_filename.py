import subprocess
from datetime import datetime
from pathlib import Path

class Filename:
    
    @staticmethod
    def generate(prefix:str):
        # Get current git commit hash
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=Path.cwd(),
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_hash = "no-git"

        # Generate timestamped filename with git hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}_{git_hash}"
        return filename

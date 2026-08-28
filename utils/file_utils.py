
import zipfile
from typing import List, Dict, Any, Optional, Tuple
import io

def create_zip_bundle(files_dict: Dict[str, str]) -> bytes:
    """Create a ZIP bundle from a dictionary of filename: content pairs"""
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in files_dict.items():
            zip_file.writestr(filename, content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()
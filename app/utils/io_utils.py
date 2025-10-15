import tempfile
import os

def tmp_path(suffix: str = "") -> str:
    """
    임시 파일 경로를 생성하는 함수
    예: tmp_path(".csv") → /tmp/tmpabcd1234.csv
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path

# utils.py
"""
공통 유틸리티 모듈
- IMG_EXTS : 지원 이미지 확장자 집합
- Tee      : print 출력을 파일에도 복사하는 간단 Tee
"""
from __future__ import annotations
import atexit
import sys
from pathlib import Path

IMG_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp"}


class Tee:
    """stdout 을 지정 파일에도 동시에 복사"""
    def __init__(self, logfile: Path):
        self.file = logfile.open("w", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self
        atexit.register(self.close)

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()

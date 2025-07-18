"""
HWP OLE 스트림 레벨에서 BinData 스트림을 외부 이미지로 교체하는 스크립트
  • 원본 포맷 및 해상도를 유지(또는 맞춤)
  • raw deflate(wbits=-MAX_WBITS) 로 재압축
  • 새 압축이 원본보다 작으면 0x00 패딩, 클 때 경고
"""
import shutil
import zlib
import olefile
import os
from io import BytesIO
from PIL import Image

# 하드코딩 매핑: 추출 키(예: '044_BinData_BIN002B.bmp') -> 교체할 이미지 경로
MAPPING = {
    '044_BinData_BIN002B.bmp': r'target_data/자동등록 사진 모음\1. 냉동기\1.jpg',
    '045_BinData_BIN002C.bmp': r'target_data/자동등록 사진 모음\1. 냉동기\2.jpg',
    '048_BinData_BIN002F.bmp': r'target_data/자동등록 사진 모음\1. 냉동기\3.jpg',
}

# 원본/결과 파일 경로
SRC_HWP = r'target_data/테스트_문서.hwp'
DST_HWP = r'target_data/테스트_문서_수정본_stream.hwp'


def replace_streams(src: str, dst: str, mapping: dict[str, str]) -> None:
    # 1) 원본 파일 복사
    shutil.copy2(src, dst)
    print(f"[INFO] 복사된 파일: {dst}")

    # 2) OLE Compound Document 열기 (쓰기 모드)
    ole = olefile.OleFileIO(dst, write_mode=True)

    # 3) 내부 스트림 목록 출력 (디버그)
    all_streams = ole.listdir(streams=True)
    print("[DEBUG] 내부 스트림 목록:")
    for path in all_streams:
        print("  -", path)

    # 4) 매핑된 스트림별로 교체
    for key, img_path in mapping.items():
        print(f"\n[STEP] 처리 매핑 키: {key}, 이미지 경로: {img_path}")
        parts = key.split('_', 2)
        leaf = parts[2] if len(parts) == 3 else key
        print(f"[DEBUG] 찾아야 할 스트림 leaf: {leaf}")

        # 스트림 경로 찾기
        target_path = next(
            (p for p in all_streams
             if p[0] == 'BinData' and p[1].upper() == leaf.upper()),
            None
        )
        if not target_path:
            print(f"[WARN] 스트림 없음: {leaf}")
            continue
        print(f"[INFO] 스트림 발견: {target_path}")

        # 4-1) 기존 스트림 데이터 읽기
        orig_comp = ole.openstream(target_path).read()
        orig_size = len(orig_comp)
        print(f"[DEBUG] 기존 압축 데이터 크기: {orig_size} bytes")

        # 4-2) 압축 해제 (raw deflate)
        try:
            raw = zlib.decompress(orig_comp, -zlib.MAX_WBITS)
            print(f"[DEBUG] raw deflate 해제 성공, raw 크기: {len(raw)} bytes")
        except zlib.error as e:
            print(f"[ERROR] 원본 스트림 압축 해제 실패: {e}")
            continue

        # 4-2-1) 원본 이미지 해상도 확인
        orig_img = Image.open(BytesIO(raw))
        orig_w, orig_h = orig_img.size
        print(f"[DEBUG] 원본 이미지 해상도: {orig_w}×{orig_h}")

        # 4-3) 새 이미지 읽어서 리사이즈
        if not os.path.isfile(img_path):
            print(f"[ERROR] 이미지 파일 없음: {img_path}")
            continue
        mapping_img = Image.open(img_path)
        resized = mapping_img.convert(orig_img.mode).resize((orig_w, orig_h))
        print(f"[DEBUG] 매핑 이미지 크기 조정: {mapping_img.size} → {(orig_w, orig_h)}")

        # 4-4) 포맷 결정 및 raw 추출
        bio = BytesIO()
        if leaf.lower().endswith('.bmp'):
            resized.save(bio, format='BMP')
            print(f"[DEBUG] BMP 변환 raw 크기: {bio.tell()} bytes")
        else:
            fmt = orig_img.format or mapping_img.format or 'PNG'
            resized.save(bio, format=fmt)
            print(f"[DEBUG] {fmt} 변환 raw 크기: {bio.tell()} bytes")
        new_raw = bio.getvalue()

        # 4-5) raw deflate 재압축
        comp_obj = zlib.compressobj(
            level=9,                  # 최대 압축
            method=zlib.DEFLATED,
            wbits=-zlib.MAX_WBITS     # raw deflate
        )
        new_comp = comp_obj.compress(new_raw) + comp_obj.flush()
        print(f"[DEBUG] raw deflate 압축 후 크기: {len(new_comp)} bytes (원본: {orig_size} bytes)")

        # ★ 4-5-5) 패딩 적용: new_comp이 작으면 0x00으로 채워서 orig_size와 동일하게
        if len(new_comp) < orig_size:
            pad_len = orig_size - len(new_comp)
            new_comp += b'\x00' * pad_len
            print(f"[DEBUG] 패딩 적용: {pad_len} bytes, 최종 크기: {len(new_comp)} bytes")
        elif len(new_comp) > orig_size:
            print(f"[WARN] 새 압축 크기({len(new_comp)})가 원본보다 큽니다. 추가 품질/해상도 조정이 필요합니다.")

        # 4-6) 스트림 덮어쓰기
        ole.write_stream(target_path, new_comp)
        print(f"[INFO] 스트림 {target_path} 덮어쓰기 완료")

    # 5) 저장 및 종료
    ole.close()
    print('[INFO] 스트림 레벨 교체 완료')


if __name__ == '__main__':
    replace_streams(SRC_HWP, DST_HWP, MAPPING)

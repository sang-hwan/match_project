#!/usr/bin/env python3
"""
HWP OLE 스트림 레벨에서 BinData 스트림을 외부 이미지로 교체하는 스크립트
  • 원본 포맷 및 해상도를 유지(또는 맞춤)
  • raw deflate(wbits=-MAX_WBITS) 로 재압축
  • JPEG일 경우 품질(quality)을 동적으로 낮춰 원본 크기 이하로 맞춤
  • PNG일 경우 optimize=True 옵션 적용
  • 최종 압축이 원본보다 작으면 0x00 패딩, 클 경우 안전한 트리밍 후 디컴프레션 검증
"""

import shutil
import zlib
import olefile
import os
import json
from io import BytesIO
from PIL import Image
import argparse


def parse_src_path(raw: str) -> str:
    """
    raw: 'target_data_자동등록 사진 모음_2. 냉각탑_20_jpg.png'
    return: 'target_data/자동등록 사진 모음/2. 냉각탑/20.jpg'
    """
    try:
        raw = raw.strip()
        p0, p1, rest = raw.split('_', 2)
        base_folder = f"{p0}_{p1}"
        album, rest2 = rest.split('_', 1)
        subfolder, file_ext = rest2.split('_', 1)
        file_base, _container_ext = file_ext.split('.', 1)
        file_num, orig_ext = file_base.split('_', 1)
        return os.path.join(
            base_folder,
            album,
            subfolder,
            f"{file_num}.{orig_ext}"
        )
    except Exception:
        print(f"[WARNING] parse_src_path: unexpected format, using raw: {raw}")
        return raw


def safe_decompressable(data: bytes) -> bool:
    """
    잘린 DEFLATE 데이터가 디컴프레션 가능한지 확인
    """
    try:
        zlib.decompress(data, -zlib.MAX_WBITS)
        return True
    except zlib.error:
        return False


def replace_streams(src: str, dst: str, mapping: dict[str, str]) -> None:
    shutil.copy2(src, dst)
    print(f"[INFO] 복사된 파일: {dst}")
    ole = olefile.OleFileIO(dst, write_mode=True)

    all_streams = ole.listdir(streams=True)
    print("[DEBUG] 내부 스트림 목록:")
    for path in all_streams:
        print("  -", path)

    for leaf, orig_img_path in mapping.items():
        print(f"\n[STEP] 스트림 leaf: {leaf}, 교체할 이미지 경로: {orig_img_path}")
        target_path = next(
            (p for p in all_streams if p[0] == 'BinData' and p[1].upper() == leaf.upper()),
            None
        )
        if not target_path:
            print(f"[WARN] 해당 스트림 없음: {leaf}")
            continue
        print(f"[INFO] 스트림 발견: {target_path}")

        orig_comp = ole.openstream(target_path).read()
        orig_size = len(orig_comp)
        print(f"[DEBUG] 기존 압축 데이터 크기: {orig_size} bytes")

        try:
            raw = zlib.decompress(orig_comp, -zlib.MAX_WBITS)
            print(f"[DEBUG] raw deflate 해제 성공, raw 크기: {len(raw)} bytes")
        except zlib.error:
            try:
                raw = zlib.decompress(orig_comp)
                print(f"[DEBUG] zlib wrapper 해제 성공, raw 크기: {len(raw)} bytes")
            except zlib.error as e:
                print(f"[ERROR] 압축 해제 실패: {e}")
                continue

        orig_img = Image.open(BytesIO(raw))
        orig_w, orig_h = orig_img.size
        print(f"[DEBUG] 원본 이미지 해상도: {orig_w}×{orig_h}")

        if not os.path.isfile(orig_img_path):
            print(f"[ERROR] 이미지 파일 없음: {orig_img_path}")
            continue
        mapping_img = Image.open(orig_img_path)
        resized = mapping_img.convert(orig_img.mode).resize((orig_w, orig_h))
        print(f"[DEBUG] 매핑 이미지 크기 조정: {mapping_img.size} → {(orig_w, orig_h)}")

        fmt = (orig_img.format or mapping_img.format or 'PNG').upper()
        new_raw = None
        new_comp = None

        # JPEG: 동적 품질 조정
        if fmt in ['JPEG', 'JPG']:
            for quality in range(95, 9, -5):
                bio = BytesIO()
                resized.save(bio, format='JPEG', quality=quality)
                candidate = bio.getvalue()
                comp_obj = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
                comp_try = comp_obj.compress(candidate) + comp_obj.flush()
                print(f"[DEBUG] JPEG quality={quality}, 압축 크기={len(comp_try)} bytes")
                if len(comp_try) <= orig_size:
                    new_raw, new_comp = candidate, comp_try
                    print(f"[INFO] 선택된 JPEG quality={quality}, 크기 맞춤 성공")
                    break
            if new_comp is None:
                # 최소 quality 에서도 크기가 크면 마지막 결과 사용
                new_raw, new_comp = candidate, comp_try
                print(f"[WARN] 모든 품질에서 크기 초과, 마지막 quality={quality} 사용")

        # PNG: 최적화 옵션 사용
        elif fmt == 'PNG':
            bio = BytesIO()
            resized.save(bio, format='PNG', optimize=True)
            new_raw = bio.getvalue()
            comp_obj = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
            new_comp = comp_obj.compress(new_raw) + comp_obj.flush()
            print(f"[DEBUG] PNG(optimize) 압축 후 크기={len(new_comp)} bytes")

        # 기타 포맷: 기본 재압축
        else:
            bio = BytesIO()
            resized.save(bio, format=fmt)
            new_raw = bio.getvalue()
            comp_obj = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
            new_comp = comp_obj.compress(new_raw) + comp_obj.flush()
            print(f"[DEBUG] 기타 포맷({fmt}) 압축 후 크기={len(new_comp)} bytes")

        comp_size = len(new_comp)

        # 패딩 또는 안전한 트리밍
        if comp_size < orig_size:
            pad_len = orig_size - comp_size
            new_comp += b'\x00' * pad_len
            print(f"[DEBUG] 패딩 적용: {pad_len} bytes, 최종 크기={len(new_comp)} bytes")

        elif comp_size > orig_size:
            trimmed = new_comp[:orig_size]
            print(f"[WARN] 크기 초과({comp_size}->{orig_size} bytes), 안전 트리밍 시도 중...")
            # 트리밍 후 디컴프레션 검증
            if safe_decompressable(trimmed):
                new_comp = trimmed
                print(f"[INFO] 트리밍 검증 성공, 최종 크기={len(new_comp)} bytes")
            else:
                # 검증 실패 시 추가 안내
                print("[ERROR] 트리밍 후에도 디컴프레션 실패: JPEG quality 또는 해상도를 더 낮춰야 합니다.")
                new_comp = trimmed  # 최후 선택

        ole.write_stream(target_path, new_comp)
        print(f"[INFO] 스트림 {target_path} 덮어쓰기 완료, 크기: {len(new_comp)} bytes")

    ole.close()
    print('[INFO] 스트림 교체 완료')


if __name__ == '__main__':
    pa = argparse.ArgumentParser(
        description='HWP BinData 스트림을 매핑 정보에 따라 교체합니다.'
    )
    pa.add_argument('--map-json', '-m', required=True, help='mapping JSON 파일')
    pa.add_argument('--src-hwp', '-s', required=True, help='원본 HWP 파일 경로')
    pa.add_argument('--dst-hwp', '-d', required=True, help='대상 HWP 파일 경로')
    args = pa.parse_args()

    raw_map = json.load(open(args.map_json, 'r', encoding='utf-8'))
    original_map: dict[str, str] = {}
    for k, v in raw_map.items():
        if 'BinData' in k:
            hwp_path, orig_path = k, v
        elif 'BinData' in v:
            hwp_path, orig_path = v, k
        else:
            hwp_path, orig_path = k, v
        fname = os.path.basename(hwp_path)
        body = fname
        if '_BinData_' in body:
            body = body.split('_BinData_', 1)[1]
        if body.lower().endswith('.png'):
            body = body[:-4]
        if '_' in body:
            base, ext = body.rsplit('_', 1)
            leaf = f"{base}.{ext}"
        else:
            leaf = body
        if os.path.exists(orig_path):
            real_orig = orig_path
        else:
            real_orig = parse_src_path(orig_path)
        original_map[leaf] = real_orig

    replace_streams(args.src_hwp, args.dst_hwp, original_map)

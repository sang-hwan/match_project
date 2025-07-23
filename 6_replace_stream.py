# 6_replace_stream.py
"""
HWP OLE 스트림 레벨에서 BinData 스트림을 외부 이미지로 교체하는 스크립트
  • 원본 포맷(raw JPEG/PNG 포함) 및 해상도 유지(필요시 축소)
  • raw deflate(wbits=-MAX_WBITS) 로 재압축
  • JPEG일 경우 품질(quality)을 동적으로 낮춰 원본 크기 이하로 맞춤
  • PNG일 경우 optimize=True 옵션 적용
  • 최종 압축이 원본보다 작으면 0x00 패딩, 클 경우 안전한 트리밍/해상도 축소 후 디컴프레션 검증
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
    """raw: 'target_data_자동…_20_jpg.png' → 'target_data/자동…/20.jpg'"""
    try:
        raw = raw.strip()
        p0, p1, rest = raw.split('_', 2)
        base_folder = f"{p0}_{p1}"
        album, rest2 = rest.split('_', 1)
        subfolder, file_ext = rest2.split('_', 1)
        file_base, _ = file_ext.split('.', 1)
        num, ext = file_base.split('_', 1)
        return os.path.join(base_folder, album, subfolder, f"{num}.{ext}")
    except Exception:
        print(f"[WARN] parse_src_path: unexpected format, using raw: {raw}")
        return raw

def safe_decompressable(data: bytes) -> bool:
    """잘린 DEFLATE 데이터가 디컴프레션 가능한지 확인"""
    try:
        zlib.decompress(data, -zlib.MAX_WBITS)
        return True
    except zlib.error:
        return False

def compress_image(img: Image.Image, fmt: str, orig_size: int):
    """
    주어진 PIL 이미지에 대해 fmt 포맷으로 재압축 시도.
    JPEG이면 품질 루프, PNG면 optimize, 기타는 기본 재압축.
    (new_raw, new_comp) 반환.
    """
    if fmt in ('JPEG', 'JPG'):
        last_candidate = last_comp = None
        for q in range(95, 9, -5):
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=q)
            data = buf.getvalue()
            comp = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
            comp_data = comp.compress(data) + comp.flush()
            print(f"[DEBUG] JPEG quality={q}, 압축 크기={len(comp_data)} bytes")
            if len(comp_data) <= orig_size:
                print(f"[INFO] 선택된 JPEG quality={q}, 크기 맞춤 성공")
                return data, comp_data
            last_candidate, last_comp = data, comp_data
        print(f"[WARN] 모든 품질에서 크기 초과, 마지막 quality 사용")
        return last_candidate, last_comp

    elif fmt == 'PNG':
        buf = BytesIO()
        img.save(buf, format='PNG', optimize=True)
        data = buf.getvalue()
        comp = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        comp_data = comp.compress(data) + comp.flush()
        print(f"[DEBUG] PNG(optimize) 압축 후 크기={len(comp_data)} bytes")
        return data, comp_data

    else:
        buf = BytesIO()
        img.save(buf, format=fmt)
        data = buf.getvalue()
        comp = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        comp_data = comp.compress(data) + comp.flush()
        print(f"[DEBUG] 기타 포맷({fmt}) 압축 후 크기={len(comp_data)} bytes")
        return data, comp_data

def replace_streams(src: str, dst: str, mapping: dict[str, str]) -> None:
    total = replaced = decompress_fail = trim_fail = 0

    shutil.copy2(src, dst)
    print(f"[INFO] 복사된 파일: {dst}")

    ole = olefile.OleFileIO(dst, write_mode=True)
    all_streams = ole.listdir(streams=True)
    print("[DEBUG] 내부 스트림 목록:")
    for p in all_streams:
        print("  -", p)

    for leaf, orig_img_path in mapping.items():
        total += 1
        print(f"\n[STEP] 스트림 leaf: {leaf}, 교체할 이미지 경로: {orig_img_path}")
        target = next((p for p in all_streams
                       if p[0]=='BinData' and p[1].upper()==leaf.upper()), None)
        if not target:
            print(f"[WARN] 해당 스트림 없음: {leaf}")
            continue
        print(f"[INFO] 스트림 발견: {target}")

        orig_comp = ole.openstream(target).read()
        orig_size = len(orig_comp)
        print(f"[DEBUG] 기존 압축 데이터 크기: {orig_size} bytes")

        # RAW JPEG/PNG 스트림은 압축 해제 없이 원본 raw 사용
        if leaf.lower().endswith(('.jpg','.jpeg','.png')):
            print("[INFO] RAW 포맷 스트림으로 간주, 압축 해제 없이 사용")
            raw = orig_comp
        else:
            try:
                raw = zlib.decompress(orig_comp, -zlib.MAX_WBITS)
                print(f"[DEBUG] raw deflate 해제 성공, 크기: {len(raw)} bytes")
            except zlib.error:
                try:
                    raw = zlib.decompress(orig_comp)
                    print(f"[DEBUG] zlib wrapper 해제 성공, 크기: {len(raw)} bytes")
                except zlib.error as e:
                    print(f"[ERROR] 압축 해제 실패: {e}")
                    decompress_fail += 1
                    continue

        # 원본 이미지 열기
        try:
            orig_img = Image.open(BytesIO(raw))
            orig_w, orig_h = orig_img.size
            print(f"[DEBUG] 원본 이미지 해상도: {orig_w}×{orig_h}")
        except Exception as e:
            print(f"[ERROR] raw 이미지 열기 실패: {e}")
            decompress_fail += 1
            continue

        # 매핑 이미지 로드 & 리사이즈
        if not os.path.isfile(orig_img_path):
            print(f"[ERROR] 이미지 파일 없음: {orig_img_path}")
            continue
        mapping_img = Image.open(orig_img_path)
        resized = mapping_img.convert(orig_img.mode).resize((orig_w,orig_h))
        print(f"[DEBUG] 매핑 이미지 크기 조정: {mapping_img.size} → {(orig_w,orig_h)}")

        fmt = (orig_img.format or mapping_img.format or 'PNG').upper()
        new_raw, new_comp = compress_image(resized, fmt, orig_size)
        comp_size = len(new_comp)

        # 패딩
        if comp_size < orig_size:
            pad = orig_size - comp_size
            new_comp += b'\x00' * pad
            print(f"[DEBUG] 패딩 적용: {pad} bytes, 최종 크기={len(new_comp)} bytes")

        # 트리밍
        if len(new_comp) > orig_size:
            trimmed = new_comp[:orig_size]
            print(f"[WARN] 크기 초과({len(new_comp)}->{orig_size} bytes), 안전 트리밍 시도...")
            if safe_decompressable(trimmed):
                new_comp = trimmed
                print(f"[INFO] 트리밍 검증 성공, 최종 크기={len(new_comp)} bytes")
            else:
                # 해상도 축소 후 재시도
                print("[WARN] 트리밍 실패, 해상도 축소 후 재시도합니다.")
                success = False
                for scale in (0.9,0.8,0.7,0.6,0.5):
                    w2 = int(orig_w*scale); h2 = int(orig_h*scale)
                    r2 = mapping_img.convert(orig_img.mode).resize((w2,h2))
                    raw2, comp2 = compress_image(r2, fmt, orig_size)
                    if len(comp2) <= orig_size and safe_decompressable(comp2):
                        # 패딩
                        if len(comp2) < orig_size:
                            comp2 += b'\x00'*(orig_size-len(comp2))
                        new_comp = comp2
                        print(f"[INFO] 해상도 {int(scale*100)}% 축소 후 최적화 성공, 최종 크기={len(new_comp)} bytes")
                        success = True
                        break
                if not success:
                    print("[ERROR] 해상도 축소 후에도 실패: 스트림 스킵")
                    trim_fail += 1
                    continue

        # 스트림 덮어쓰기
        ole.write_stream(target, new_comp)
        print(f"[INFO] 스트림 {target} 덮어쓰기 완료, 크기: {len(new_comp)} bytes")
        replaced += 1

    ole.close()
    # 요약
    print(f"\n[SUMMARY] 전체스트림={total}, 교체성공={replaced}, "
          f"압축해제실패={decompress_fail}, 트리밍실패={trim_fail}")

if __name__ == '__main__':
    pa = argparse.ArgumentParser(
        description='HWP BinData 스트림을 매핑 정보에 따라 교체합니다.'
    )
    pa.add_argument('--map-json','-m', required=True, help='mapping JSON 파일')
    pa.add_argument('--src-hwp','-s', required=True, help='원본 HWP 파일 경로')
    pa.add_argument('--dst-hwp','-d', required=True, help='대상 HWP 파일 경로')
    args = pa.parse_args()

    raw_map = json.load(open(args.map_json,'r',encoding='utf-8'))
    original_map: dict[str,str] = {}
    for k,v in raw_map.items():
        if 'BinData' in k:
            h,v2 = k,v
        elif 'BinData' in v:
            h,v2 = v,k
        else:
            h,v2 = k,v
        fname = os.path.basename(h)
        body = fname.split('_BinData_',1)[1] if '_BinData_' in fname else fname
        leaf = (body[:-4] if body.lower().endswith('.png') else body)
        if '_' in leaf:
            b, e = leaf.rsplit('_',1); leaf = f"{b}.{e}"
        real = v2 if os.path.exists(v2) else parse_src_path(v2)
        original_map[leaf] = real

    replace_streams(args.src_hwp, args.dst_hwp, original_map)

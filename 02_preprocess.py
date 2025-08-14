"""
02_preprocess.py  (rev. feedback-loop ready)

CPU-only 전처리 파이프라인
- 오토 트림(가장 큰 컨투어 → 실패 시 원본 유지, --auto-trim-plus로 보정 강화)
- 리사이즈: max(side)={--low-size, --high-size}, letterbox(+--pad-multiple 정렬)
- 색 보정(옵션): LAB-CLAHE(L), Gray-CLAHE
- 산출: {category}/{track}/{channel}/...  (category: extracted|reference, track: low|high, channel: color|gray)
- 매니페스트: preprocess_mapping.json  (03/04/05와 호환 유지)
- 증분 실행: --feedback-in / --only-refs / --only-refs-from-feedback 로 **부분 재생성** 지원
- 안전성: Windows 비ASCII 경로 I/O(imdecode/imencode), 존재 파일 건너뛰기(--skip-existing)

Usage
-----
python 02_preprocess.py EXTRACTED_IN EXTRACTED_OUT REFERENCE_IN REFERENCE_OUT \
  --low-size 640 --enable-high --high-size 1280 \
  --pad-mode edge --pad-multiple 16 \
  --colorspace lab --clahe-clip 2.0 --clahe-grid 8 \
  --gray-clahe --out-ext png \
  --map-json artifacts/preprocess_mapping.json \
  [--feedback-in artifacts/feedback.json --only-refs-from-feedback] \
  [--only-refs refs.txt --only-category reference] \
  [--auto-trim-plus] [--skip-existing]

Notes
-----
- 출력 매핑(JSON) 스키마는 기존 03_candidates.py / 04_match.py / 05_eval_inspect.py가 기대하는 형태를 그대로 유지합니다.
- --only-refs* 옵션이 없으면 **전체**를 처리합니다. 옵션이 있으면 해당 대상 **만** 재생성하고, 기존 매핑과 **병합(update)** 합니다.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:
    raise SystemExit(f"[ERR] OpenCV(cv2) import 실패: {e}\n설치: pip install opencv-python")

# =========================
# ---- 경로/이미지 I/O ----
# =========================

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")

def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    files.sort()
    return files

def imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None

def imwrite_unicode(path: Path, img: np.ndarray, ext: str,
                    jpeg_quality: int = 95, png_compress: int = 3) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ext = ext.lower()
        if ext in (".jpg", ".jpeg"):
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        elif ext == ".png":
            ok, enc = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compress)])
        elif ext == ".webp":
            ok, enc = cv2.imencode(".webp", img, [])
        else:
            ok, enc = cv2.imencode(ext, img, [])
        if not ok:
            return False
        enc.tofile(str(path))
        return True
    except Exception:
        return False

# =========================
# ---- 전처리 연산들  ----
# =========================

def to_bgr(img: np.ndarray) -> np.ndarray:
    """BGRA/GRAY 등 다양한 입력을 안전하게 BGR로 정규화."""
    if img is None:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def auto_trim(img_bgr: np.ndarray,
              method: str = "auto",
              canny1: int = 50,
              canny2: int = 150,
              min_area_ratio: float = 0.2,
              margin_ratio: float = 0.01,
              plus: bool = False) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """
    가장 큰 컨투어를 찾아 콘텐츠 영역으로 크롭. 실패 시 원본 반환.
    Returns: (cropped_img, (top, bottom, left, right) removed)
    """
    h, w = img_bgr.shape[:2]
    if method != "auto":
        return img_bgr, (0,0,0,0)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 조명 보정용 블러
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, int(canny1), int(canny2))

    if plus:
        # 경계 연결을 돕기 위한 팽창/수축(Closing)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
        # threshold 혼합 시도
        thr = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1]
        edges = cv2.bitwise_or(edges, thr)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr, (0,0,0,0)

    cnt = max(cnts, key=cv2.contourArea)
    x,y,wc,hc = cv2.boundingRect(cnt)
    area_ratio = (wc*hc) / float(w*h)

    # plus 모드에서는 문턱을 완화
    min_area = min_area_ratio * (0.5 if plus else 1.0)
    margin = margin_ratio * (2.0 if plus else 1.0)

    if area_ratio < min_area:
        return img_bgr, (0,0,0,0)

    # 마진 고려한 안전 크롭
    dx = int(round(wc * margin))
    dy = int(round(hc * margin))
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(w, x + wc + dx)
    y1 = min(h, y + hc + dy)
    cropped = img_bgr[y0:y1, x0:x1].copy()

    top = y0
    bottom = h - y1
    left = x0
    right = w - x1
    return cropped, (top, bottom, left, right)

def resize_letterbox(img_bgr: np.ndarray,
                     max_side: int,
                     pad_mode: str = "edge",
                     pad_multiple: int = 16,
                     constant_color: Tuple[int,int,int]=(114,114,114)) -> Tuple[np.ndarray, float, Tuple[int,int,int,int]]:
    """
    긴 변을 max_side로 맞추고 종횡비 유지. pad_multiple 배수로 패딩.
    Returns: (padded_img, scale, (top,bottom,left,right))
    """
    h, w = img_bgr.shape[:2]
    if max(h,w) <= 0:
        return img_bgr, 1.0, (0,0,0,0)

    scale = float(max_side) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # pad_multiple에 맞춰 최소 패딩
    pad_w = (int(np.ceil(new_w / pad_multiple)) * pad_multiple) - new_w
    pad_h = (int(np.ceil(new_h / pad_multiple)) * pad_multiple) - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    border_type = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "none": cv2.BORDER_CONSTANT,  # 사용 안 함
    }.get(pad_mode, cv2.BORDER_REPLICATE)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, border_type, value=constant_color)
    return padded, scale, (top, bottom, left, right)

def enhance_color_lab(img_bgr: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def to_gray(img_bgr: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g = clahe.apply(g)
    return g

# =========================
# ---- 매핑 레코드  ----
# =========================

@dataclass
class VariantRecord:
    category: str          # "extracted" | "reference"
    src_abspath: str
    src_relpath: str
    track: str             # "low" | "high"
    channel: str           # "color" | "gray"
    out_abspath: str
    out_relpath: str
    orig_wh: Tuple[int,int]                 # (W0,H0)
    trimmed_pad: Tuple[int,int,int,int]     # (top,bottom,left,right) removed
    scale: float
    pad: Tuple[int,int,int,int]             # (top,bottom,left,right) added

# =========================
# ---- 대상 결정 로직 ----
# =========================

def _normalize_key(s: str) -> str:
    """키 표준화: 'low|', 'high|' 접두어 제거하고 슬래시 통일"""
    s = s.strip().replace("\\", "/")
    if "|" in s:
        pref, rest = s.split("|", 1)
        if pref in ("low", "high"):
            return rest
    return s

def load_feedback_targets(path: Optional[Path]) -> Dict[str, Set[str]]:
    """
    feedback.json에서 재전처리 대상(src_relpath) 집합을 추출.
    Returns: {"reference": set(), "extracted": set()}
    지원 포맷:
      actions.redo_preprocess.refs            -> reference
      actions.redo_preprocess.extracted_refs  -> extracted (있을 때만)
    """
    targets = {"reference": set(), "extracted": set()}
    if not path:
        return targets
    if not path.exists():
        print(f"[WARN] feedback 파일 없음: {path}")
        return targets
    try:
        with open(path, "r", encoding="utf-8") as f:
            fb = json.load(f)
        actions = fb.get("actions", {})
        rp = actions.get("redo_preprocess", {})
        for k, cat in (("refs", "reference"), ("extracted_refs", "extracted")):
            lst = rp.get(k) or []
            for s in lst:
                targets[cat].add(_normalize_key(str(s)))
        return targets
    except Exception as e:
        print(f"[WARN] feedback 파싱 실패: {e}")
        return targets

def load_only_refs(path: Optional[Path]) -> Set[str]:
    """텍스트 파일(한 줄당 하나)에서 src_relpath 집합을 로드."""
    out: Set[str] = set()
    if not path:
        return out
    if not path.exists():
        print(f"[WARN] only-refs 파일 없음: {path}")
        return out
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.add(_normalize_key(line))
    except Exception as e:
        print(f"[WARN] only-refs 파싱 실패: {e}")
    return out

def should_process_this(relpath: Path,
                        category: str,
                        args,
                        fb_targets: Dict[str, Set[str]],
                        only_refs: Set[str]) -> bool:
    """선택적 부분 처리 로직."""
    if args.only_category != "both" and category != args.only_category:
        return False
    if args.only_refs_from_feedback and fb_targets.get(category):
        return relpath.as_posix() in fb_targets[category]
    if only_refs:
        return relpath.as_posix() in only_refs
    # 아무 제한이 없으면 전체 처리
    return True

def outputs_exist(out_root: Path, rel: Path, track: str, ext: str) -> bool:
    """color/gray 두 파일이 모두 존재하면 존재로 간주."""
    color = out_root / track / "color" / (str(rel.with_suffix("")) + f".{ext}")
    gray  = out_root / track / "gray"  / (str(rel.with_suffix("")) + f".{ext}")
    return color.exists() and gray.exists()

# =========================
# ---- 메인 처리 루틴 ----
# =========================

def process_one_image(category: str,
                      src_path: Path,
                      src_root: Path,
                      out_root: Path,
                      args) -> List[VariantRecord]:
    recs: List[VariantRecord] = []

    img0 = imread_unicode(src_path)
    if img0 is None:
        print(f"[WARN] 읽기 실패: {src_path}")
        return recs

    img0 = to_bgr(img0)
    H0, W0 = img0.shape[:2]
    rel = src_path.relative_to(src_root)

    # 1) 트림
    if args.trim == "auto":
        img1, removed = auto_trim(
            img0, "auto",
            args.canny1, args.canny2,
            args.min_area_ratio, args.margin_ratio,
            plus=args.auto_trim_plus
        )
    else:
        img1, removed = img0, (0,0,0,0)

    # 2) 색/회색 변환 원본
    color_base = enhance_color_lab(img1, args.clahe_clip, args.clahe_grid) if args.colorspace == "lab" else img1
    gray_base  = to_gray(color_base, use_clahe=args.gray_clahe)

    def process_track(track_name: str, max_side: int):
        nonlocal recs
        if args.skip_existing and outputs_exist(out_root, rel, track_name, args.out_ext):
            # 건너뛰기(매핑은 기존 것을 유지/병합 단계에서 보존)
            return

        resized, scale, pad = resize_letterbox(color_base, max_side, args.pad_mode, args.pad_multiple)
        color = resized
        gray  = to_gray(resized, use_clahe=args.gray_clahe)

        stem = rel.with_suffix("")  # keep subdirs, remove ext
        out_color = out_root / track_name / "color" / (str(stem) + f".{args.out_ext}")
        out_gray  = out_root / track_name / "gray"  / (str(stem) + f".{args.out_ext}")

        ok1 = imwrite_unicode(out_color, color, f".{args.out_ext}", args.jpeg_q, args.png_c)
        ok2 = imwrite_unicode(out_gray,  gray,  f".{args.out_ext}", args.jpeg_q, args.png_c)

        if ok1:
            recs.append(VariantRecord(
                category=category,
                src_abspath=str(src_path.resolve()),
                src_relpath=str(rel).replace("\\", "/"),
                track=track_name,
                channel="color",
                out_abspath=str(out_color.resolve()),
                out_relpath=str(out_color.relative_to(out_root)).replace("\\", "/"),
                orig_wh=(W0, H0),
                trimmed_pad=removed,
                scale=scale,
                pad=pad
            ))
        else:
            print(f"[WARN] 저장 실패(color): {out_color}")
        if ok2:
            recs.append(VariantRecord(
                category=category,
                src_abspath=str(src_path.resolve()),
                src_relpath=str(rel).replace("\\", "/"),
                track=track_name,
                channel="gray",
                out_abspath=str(out_gray.resolve()),
                out_relpath=str(out_gray.relative_to(out_root)).replace("\\", "/"),
                orig_wh=(W0, H0),
                trimmed_pad=removed,
                scale=scale,
                pad=pad
            ))
        else:
            print(f"[WARN] 저장 실패(gray): {out_gray}")

    # low
    process_track("low", args.low_size)
    # high
    if args.enable_high:
        process_track("high", args.high_size)

    return recs

def process_category(category: str,
                     in_root: Path,
                     out_root: Path,
                     args,
                     fb_targets: Dict[str, Set[str]],
                     only_refs: Set[str]) -> List[VariantRecord]:
    files = list_images(in_root)
    # 선택적 부분 처리
    if args.only_category != "both" or args.only_refs_from_feedback or only_refs:
        selected = []
        for p in files:
            rel = p.relative_to(in_root).as_posix()
            if should_process_this(Path(rel), category, args, fb_targets, only_refs):
                selected.append(p)
        files = selected
    print(f"[INFO] {category}: 입력 {len(files)}개 (root={in_root})")

    all_recs: List[VariantRecord] = []
    for i, p in enumerate(files, 1):
        if args.max_files and i > args.max_files:
            break
        recs = process_one_image(category, p, in_root, out_root, args)
        all_recs.extend(recs)
        if i % 50 == 0:
            print(f"[INFO] {category}: 진행 {i}/{len(files)}")
    print(f"[INFO] {category}: 산출 변형 {len(all_recs)}개")
    return all_recs

# =========================
# ---- 매핑 저장/병합 ----
# =========================

def build_mapping(args, recs_ex: List[VariantRecord], recs_ref: List[VariantRecord]) -> Dict:
    by_src = {"extracted": {}, "reference": {}}
    for r in recs_ex + recs_ref:
        cat = r.category
        d = by_src[cat].setdefault(r.src_relpath, {
            "src_abspath": r.src_abspath,
            "variants": {"low": {}, "high": {}},
            "orig_wh": r.orig_wh,
        })
        d["orig_wh"] = r.orig_wh  # 업데이트(일관성)
        d["variants"].setdefault(r.track, {})
        d["variants"][r.track][r.channel] = r.out_abspath

    mapping = {
        "version": 1,
        "params": {
            "low_size": args.low_size,
            "enable_high": args.enable_high,
            "high_size": args.high_size,
            "pad_mode": args.pad_mode,
            "pad_multiple": args.pad_multiple,
            "colorspace": args.colorspace,
            "clahe_clip": args.clahe_clip,
            "clahe_grid": args.clahe_grid,
            "gray_clahe": args.gray_clahe,
            "trim": args.trim,
            "canny1": args.canny1,
            "canny2": args.canny2,
            "min_area_ratio": args.min_area_ratio,
            "margin_ratio": args.margin_ratio,
            "out_ext": args.out_ext,
            "jpeg_q": args.jpeg_q,
            "png_c": args.png_c,
            "auto_trim_plus": args.auto_trim_plus,
            "skip_existing": args.skip_existing,
        },
        "roots": {
            "extracted_in": str(Path(args.extracted_in).resolve()),
            "extracted_out": str(Path(args.extracted_out).resolve()),
            "reference_in": str(Path(args.reference_in).resolve()),
            "reference_out": str(Path(args.reference_out).resolve()),
        },
        "summary": {
            "extracted_files": len(set([r.src_relpath for r in recs_ex])),
            "reference_files": len(set([r.src_relpath for r in recs_ref])),
            "variants_total": len(recs_ex) + len(recs_ref),
        },
        "index": [asdict(r) for r in (recs_ex + recs_ref)],
        "by_src": by_src,
    }
    return mapping

def merge_mappings(old: Dict, new: Dict) -> Dict:
    """기존 매핑과 새 매핑을 병합. 새 레코드가 동일 src_relpath를 **덮어씀**."""
    if not old:
        return new
    merged = dict(old)  # 얕은 복사
    # by_src 병합
    merged_by_src = {"extracted": {}, "reference": {}}
    for cat in ("extracted", "reference"):
        old_cat = old.get("by_src", {}).get(cat, {})
        new_cat = new.get("by_src", {}).get(cat, {})
        merged_cat = dict(old_cat)
        for k, v in new_cat.items():
            merged_cat[k] = v  # 덮어쓰기
        merged_by_src[cat] = merged_cat
    merged["by_src"] = merged_by_src

    # index 재구성(중복 제거)
    # 키: (category, src_relpath, track, channel)
    def rec_key(d: Dict) -> Tuple[str,str,str,str]:
        return (d["category"], d["src_relpath"], d["track"], d["channel"])
    old_idx = {rec_key(x): x for x in old.get("index", [])}
    new_idx = {rec_key(x): x for x in new.get("index", [])}
    old_idx.update(new_idx)  # 새로 덮어씀
    merged["index"] = list(old_idx.values())

    # summary 재계산
    ex_paths = set()
    ref_paths = set()
    for x in merged["index"]:
        if x["category"] == "extracted":
            ex_paths.add(x["src_relpath"])
        else:
            ref_paths.add(x["src_relpath"])
    merged["summary"] = {
        "extracted_files": len(ex_paths),
        "reference_files": len(ref_paths),
        "variants_total": len(merged["index"]),
    }

    # params 최신화(새 실행 파라미터 반영), 이력 보존
    params_hist = list(old.get("params_history", []))
    if "params" in old:
        params_hist.append(old["params"])
    merged["params_history"] = params_hist
    merged["params"] = new.get("params", old.get("params", {}))
    return merged

def save_mapping(path: Path, mapping: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

# =========================
# --------- CLI ----------
# =========================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Preprocess images (trim, resize+letterbox, color/gray) and write mapping JSON (03/04/05 호환)."
    )
    ap.add_argument("extracted_in", help="추출 이미지 입력 루트")
    ap.add_argument("extracted_out", help="추출 이미지 출력 루트")
    ap.add_argument("reference_in", help="원본 이미지 입력 루트")
    ap.add_argument("reference_out", help="원본 이미지 출력 루트")

    ap.add_argument("--low-size", type=int, default=640, help="low 트랙 max(side)")
    ap.add_argument("--enable-high", action="store_true", help="high 트랙 생성 여부")
    ap.add_argument("--high-size", type=int, default=1280, help="high 트랙 max(side)")

    ap.add_argument("--pad-mode", choices=["none", "constant", "edge", "reflect"], default="edge", help="letterbox 패딩 방식")
    ap.add_argument("--pad-multiple", type=int, default=16, help="출력 크기 정렬 배수")

    ap.add_argument("--colorspace", choices=["none", "lab"], default="lab", help="color 보정 방식")
    ap.add_argument("--clahe-clip", type=float, default=2.0, help="LAB-CLAHE clipLimit")
    ap.add_argument("--clahe-grid", type=int, default=8, help="LAB-CLAHE tileGridSize")

    ap.add_argument("--gray-clahe", action="store_true", help="회색조에도 CLAHE 적용")
    ap.add_argument("--trim", choices=["auto", "none"], default="auto", help="오토 트림 사용 여부")
    ap.add_argument("--canny1", type=int, default=50, help="Canny threshold1")
    ap.add_argument("--canny2", type=int, default=150, help="Canny threshold2")
    ap.add_argument("--min-area-ratio", type=float, default=0.2, help="컨투어 박스 최소 비율")
    ap.add_argument("--margin-ratio", type=float, default=0.01, help="트림 박스 여유 비율")

    ap.add_argument("--out-ext", choices=["png", "jpg", "jpeg", "webp"], default="png", help="출력 이미지 확장자")
    ap.add_argument("--jpeg-q", type=int, default=95, help="JPG 저장 품질")
    ap.add_argument("--png-c", type=int, default=3, help="PNG 압축 레벨(0~9)")

    # --- 신규(루프/증분 실행용) ---
    ap.add_argument("--feedback-in", default=None, help="05단계 생성 feedback.json 경로")
    ap.add_argument("--only-refs-from-feedback", action="store_true", help="feedback.json의 redo_preprocess 대상만 처리")
    ap.add_argument("--only-refs", default=None, help="처리할 src_relpath 목록 파일(한 줄당 하나)")
    ap.add_argument("--only-category", choices=["reference", "extracted", "both"], default="both", help="처리 카테고리 제한")
    ap.add_argument("--auto-trim-plus", action="store_true", help="오토 트림 보정 강화(연결/마진 완화)")
    ap.add_argument("--skip-existing", action="store_true", help="출력 파일이 이미 있으면 재생성 생략")

    ap.add_argument("--map-json", default="preprocess_mapping.json", help="매핑 JSON 저장 경로")
    ap.add_argument("--max-files", type=int, default=0, help="디버그용: 카테고리별 최대 처리 개수(0=무제한)")

    return ap.parse_args()

def main() -> int:
    args = parse_args()

    ex_in = Path(args.extracted_in)
    ex_out = Path(args.extracted_out)
    ref_in = Path(args.reference_in)
    ref_out = Path(args.reference_out)
    map_path = Path(args.map_json)

    fb_targets = load_feedback_targets(Path(args.feedback_in)) if args.feedback_in else {"reference": set(), "extracted": set()}
    only_refs = load_only_refs(Path(args.only_refs)) if args.only_refs else set()

    print("[INFO] 전처리 시작")
    recs_ex = process_category("extracted", ex_in, ex_out, args, fb_targets, only_refs)
    recs_ref = process_category("reference", ref_in, ref_out, args, fb_targets, only_refs)

    new_mapping = build_mapping(args, recs_ex, recs_ref)

    # 기존 매핑과 병합(부분 처리 지원)
    if map_path.exists() and (args.only_refs_from_feedback or args.only_refs or args.only_category != "both"):
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            merged = merge_mappings(old, new_mapping)
            save_mapping(map_path, merged)
            print(f("[INFO] 매핑 병합 저장: {map_path}"))
        except Exception as e:
            print(f"[WARN] 기존 매핑 병합 실패, 신규로 저장합니다: {e}")
            save_mapping(map_path, new_mapping)
    else:
        save_mapping(map_path, new_mapping)
        print(f"[INFO] 매핑 저장: {map_path}")

    print(f"[INFO] 완료: extracted 변형 {len(recs_ex)}, reference 변형 {len(recs_ref)}, 총 {len(recs_ex)+len(recs_ref)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

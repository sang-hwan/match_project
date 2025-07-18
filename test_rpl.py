# test_rpl.py
"""
하드코딩된 매핑으로 HWP 문서 내 특정 이미지를 외부 파일로 교체 테스트
"""
from replace_hwp_images import replace_images

def main():
    # 원본 HWP 파일 경로
    hwp_path = r"target_data/테스트_문서.hwp"
    # 교체 후 저장할 파일 경로
    output_path = r"target_data/테스트_문서_수정본_test.hwp"
    # 하드코딩된 매핑: 문서 내 추출 이미지 번호 → 새 이미지 파일 경로
    replacements = {
        44: r"target_data/자동등록 사진 모음\1. 냉동기\1.jpg",
        45: r"target_data/자동등록 사진 모음\1. 냉동기\2.jpg",
        48: r"target_data/자동등록 사진 모음\1. 냉동기\3.jpg",
    }

    print(f"[INFO] 교체 테스트 시작: {hwp_path} → {output_path}")
    replace_images(hwp_path, replacements, output_path)


if __name__ == '__main__':
    main()

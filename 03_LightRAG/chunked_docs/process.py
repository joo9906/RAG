"""
JSONL → MD 변환 스크립트

chunked_docs/ 폴더의 JSONL 파일들을 읽어서
각 청크를 개별 .md 파일로 변환한 뒤
04_OpenKB/docs/ 폴더에 저장합니다.

파일명 규칙: {doc_prefix}_{chunk_index:04d}.md
"""

import json
import os
import re

# ── 경로 설정 ──────────────────────────────────────────────
INPUT_DIR = r"C:\Users\happy\OneDrive\바탕 화면\Code\10_RAG\03_LightRAG\chunked_docs"
OUTPUT_DIR = r"C:\Users\happy\OneDrive\바탕 화면\Code\10_RAG\04_OpenKB\docs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_filename(name: str, max_len: int = 60) -> str:
    """파일명에 사용할 수 없는 문자를 제거하고 길이를 제한합니다."""
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = name.strip()
    return name[:max_len]


def build_md(chunk: dict) -> str:
    """청크 딕셔너리를 마크다운 텍스트로 변환합니다."""
    lines = []

    # 제품명 헤더
    product_names = chunk.get("product_names", [])
    if product_names:
        lines.append(f"# {', '.join(product_names)}")
    else:
        lines.append(f"# {chunk.get('doc_name', 'Document')}")

    lines.append("")

    # 메타 정보
    lines.append(f"> **문서**: {chunk.get('doc_name', '')}")
    lines.append(f"> **청크 ID**: {chunk.get('chunk_id', '')}")
    lines.append("")

    # 본문
    text = chunk.get("text", "").strip()
    lines.append(text)

    # 참고문헌
    refs = chunk.get("refs", [])
    if refs:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 참고문헌")
        for ref in refs:
            lines.append(f"- {ref}")

    return "\n".join(lines)


# ── 변환 실행 ──────────────────────────────────────────────
total_chunks = 0
total_files = 0

for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.endswith(".jsonl"):
        continue

    jsonl_path = os.path.join(INPUT_DIR, filename)
    doc_prefix = safe_filename(os.path.splitext(filename)[0])

    # JSONL → 청크 목록
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [경고] JSON 파싱 실패: {filename} - {e}")

    if not chunks:
        print(f"[건너뜀] 청크 없음: {filename}")
        continue

    # 문서별 하위 폴더 생성
    doc_output_dir = os.path.join(OUTPUT_DIR, doc_prefix)
    os.makedirs(doc_output_dir, exist_ok=True)

    # 각 청크를 md 파일로 저장
    for idx, chunk in enumerate(chunks, start=1):
        md_content = build_md(chunk)
        md_filename = f"{doc_prefix}_{idx:04d}.md"
        md_path = os.path.join(doc_output_dir, md_filename)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

    print(f"[완료] {filename}  →  {len(chunks)}개 청크")
    total_chunks += len(chunks)
    total_files += 1

print()
print(f"=== 변환 완료: {total_files}개 JSONL → {total_chunks}개 MD 파일 ===")
print(f"저장 경로: {OUTPUT_DIR}")
from __future__ import annotations

import os
import logging
import re
from typing import Optional
from fastapi import Request


def convert_to_pdf(file_path: str, use_pdf_sdk: bool = True) -> str | None:
    """
    주어진 문서를 PDF 형식으로 변환합니다.

    :param file_path: 변환할 원본 파일의 경로
    :param use_pdf_sdk: Genon PDF SDK 사용 여부 (False인 경우 LibreOffice 사용)
    :return: 변환된 PDF 파일의 경로. 실패 시 None 반환.
    """
    from genon.preprocessor.converters.hwp_to_pdf import convert_hwp_to_pdf
    
    primary = "pdf_sdk" if use_pdf_sdk else "libreoffice"
    return convert_hwp_to_pdf(file_path, primary=primary, disable_fallback=True)


def _is_pdf(file_path: str) -> bool:
    """
    파일의 매직 헤더를 검사하여 실제 PDF 파일인지 확인합니다.

    :param file_path: 검사할 파일 경로
    :return: PDF 파일이면 True, 아니면 False
    """
    try:
        with open(file_path, "rb") as f:
            # PDF 파일은 항상 "%PDF-"로 시작함
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


class GenosServiceException(Exception):
    """
    GenOS 전처리 서비스 내에서 발생하는 커스텀 예외 클래스입니다.
    에러 코드와 메시지를 명시적으로 관리하기 위해 사용됩니다.
    """
    def __init__(self, error_code: str, error_msg: Optional[str] = None, msg_params: Optional[dict] = None) -> None:
        self.code = 1
        self.error_code = error_code
        self.error_msg = error_msg or "GenOS Service Exception"
        self.msg_params = msg_params or {}

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(code={self.code!r}, errMsg={self.error_msg!r})"


async def assert_cancelled(request: Request):
    """
    클라이언트의 요청이 취소되었는지(Disconnected) 확인하고,
    취소된 경우 예외를 발생시켜 파이프라인 처리를 중단합니다.

    :param request: FastAPI Request 객체
    """
    if await request.is_disconnected():
        raise GenosServiceException(1, f"Cancelled")


# ==============================================================================
# 시스템 및 유저 프롬프트 (목차 생성 및 메타데이터 추출용)
# ==============================================================================

toc_system_prompt = """You are an expert at generating table of contents (목차) from Korean documents. You specialize in regulatory documents, terms of service, contracts, and mixed-format documents that combine formal regulatory structures with general section headers.
""".strip()

toc_user_prompt = """
Here is the Korean document you need to analyze:
<document>
{raw_text}
</document>

Your task is to extract and organize all structural elements from this document into a hierarchical table of contents. Korean documents often have mixed structures where some sections follow formal regulatory patterns (제x장/절/관/조) while others use general section numbering and headers.

## Analysis Process
Before generating the final table of contents, work through the document systematically in `<analysis>` tags. It's OK for this section to be quite long. Follow these steps:
1. **Document Title Extraction**: Quote the main document title exactly as it appears at the beginning of the document.
2. **Structural Marker Identification**: Scan through the document and quote all the key structural markers you find, such as:
   - Formal regulatory patterns: 제x장, 제x절, 제x관, 제x조
   - General section patterns: numbered headers (1., 2., etc.), lettered headers (가., 나., etc.)
   - Special sections: 부칙, 별지, 별표, etc.
3. **Systematic Section Extraction**: Work through the document from beginning to end, extracting each structural element in order:
   - For each main section, quote the exact title as it appears
   - For each subsection, quote the exact title and note which main section it belongs under
   - For each article/item, quote the exact title and note its parent section
   - Include any appendices, attachments, and addenda
4. **Hierarchy Building**: For each extracted element, explicitly note:
   - What level it should be at (main section, subsection, sub-subsection, etc.)
   - What its parent section is (if any)
   - What numbering it should receive in the final TOC (1., 1.1., 1.1.1., etc.)
5. **Structure Verification**: Review your extracted elements to ensure:
   - All structural elements are captured in document order
   - The hierarchy makes logical sense
   - No elements are duplicated or missed

## Output Requirements
After your analysis, generate the table of contents with this exact format:
```
<toc>
TITLE:<document title>
1. <first main section title>
1.1. <first subsection title>
1.1.1. <first sub-subsection title>
1.2. <second subsection title>
2. <second main section title>
2.1. <subsection under second main section>
3. <third main section title>
</toc>
```

## Formatting Guidelines
- Start with `TITLE:` followed by the document title
- Use hierarchical decimal numbering (1, 1.1, 1.1.1, etc.)
- Follow each number with a space and the original title exactly as it appears
- Maintain the document's logical hierarchy
- Include appendices, attachments, and addenda as separate top-level items
- Extract titles exactly as they appear - do not include explanatory content
- Handle both formal regulatory structures and general section headers
- Wrap the entire table of contents in `<toc></toc>` tags
""".strip()

# ==============================================================================
# 컨텍스트 기반 청크 요약용 프롬프트 (Contextual Chunking)
# ==============================================================================

contextual_chunk_system_prompt = """You are a helpful AI assistant specialized in analyzing documents and generating concise contextual summaries for specific text chunks. Your goal is to provide a brief explanation of how a specific chunk of text fits within the broader context of the entire document.
""".strip()

contextual_chunk_user_prompt = """
<document>
{document_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short, succinct context to situate this chunk within the overall document to improve search retrieval of the chunk. Answer in Korean. Answer ONLY with the succinct context without any conversational filler or introductions.
""".strip()
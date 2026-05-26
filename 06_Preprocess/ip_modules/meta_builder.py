from __future__ import annotations

import json
from pydantic import BaseModel
from typing import Optional
from docling_core.types import DoclingDocument
from docling_core.types.doc import PictureItem


class GenOSVectorMeta(BaseModel):
    """
    청킹(Chunking)이 완료된 텍스트와 함께 Vector DB에 삽입되는 메타데이터 모델입니다.
    """
    class Config:
        extra = "allow"

    text: str = None
    n_char: int = None
    n_word: int = None
    n_line: int = None
    e_page: int = None
    i_page: int = None
    i_chunk_on_page: int = None
    n_chunk_of_page: int = None
    i_chunk_on_doc: int = None
    n_chunk_of_doc: int = None
    n_page: int = None
    reg_date: str = None
    chunk_bboxes: str = None
    media_files: str = None
    title: str = None
    created_date: int = None
    appendix: str = None
    file_path: Optional[str] = None


class GenOSVectorMetaBuilder:
    """
    GenOSVectorMeta 객체를 생성하기 위한 Builder 패턴 클래스입니다.
    메서드 체이닝(Method Chaining)을 지원하여 가독성 높게 메타데이터를 조립할 수 있습니다.
    """

    def __init__(self):
        self.text: Optional[str] = None
        self.n_char: Optional[int] = None
        self.n_word: Optional[int] = None
        self.n_line: Optional[int] = None
        self.i_page: Optional[int] = None
        self.e_page: Optional[int] = None
        self.i_chunk_on_page: Optional[int] = None
        self.n_chunk_of_page: Optional[int] = None
        self.i_chunk_on_doc: Optional[int] = None
        self.n_chunk_of_doc: Optional[int] = None
        self.n_page: Optional[int] = None
        self.reg_date: Optional[str] = None
        self.chunk_bboxes: Optional[str] = None
        self.media_files: Optional[str] = None
        self.title: Optional[str] = None
        self.created_date: Optional[int] = None
        self.appendix: Optional[str] = None
        self.file_path: Optional[str] = None

    def set_text(self, text: str) -> "GenOSVectorMetaBuilder":
        """청크의 원본 텍스트 및 관련 통계(문자수, 단어수, 라인수)를 설정합니다."""
        self.text = text
        self.n_char = len(text)
        self.n_word = len(text.split())
        self.n_line = len(text.splitlines())
        return self

    def set_page_info(self, i_page: int, i_chunk_on_page: int, n_chunk_of_page: int) -> "GenOSVectorMetaBuilder":
        """해당 청크가 위치한 페이지 정보 및 페이지 내 순서를 설정합니다."""
        self.i_page = i_page
        self.i_chunk_on_page = i_chunk_on_page
        self.n_chunk_of_page = n_chunk_of_page
        return self

    def set_chunk_index(self, i_chunk_on_doc: int) -> "GenOSVectorMetaBuilder":
        """문서 전체 기준 해당 청크의 순번(Index)을 설정합니다."""
        self.i_chunk_on_doc = i_chunk_on_doc
        return self

    def set_global_metadata(self, **global_metadata) -> "GenOSVectorMetaBuilder":
        """문서 전체에서 공통으로 적용되는 전역 메타데이터(생성일, 제목 등)를 동적으로 주입합니다."""
        for key, value in global_metadata.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def set_chunk_bboxes(self, doc_items: list, document: DoclingDocument) -> "GenOSVectorMetaBuilder":
        """
        청크에 포함된 Docling 문서 아이템들의 바운딩 박스(Bounding Box) 정보들을 추출하고
        정규화하여 JSON 배열 형태의 문자열로 저장합니다.
        """
        chunk_bboxes = []
        for item in doc_items:
            for prov in item.prov:
                label = item.self_ref
                type_ = item.label
                size = document.pages.get(prov.page_no).size
                page_no = prov.page_no
                bbox = prov.bbox
                
                # 좌표를 0~1 사이로 정규화
                bbox_data = {
                    "l": bbox.l / size.width,
                    "t": bbox.t / size.height,
                    "r": bbox.r / size.width,
                    "b": bbox.b / size.height,
                    "coord_origin": bbox.coord_origin.value,
                }
                
                chunk_bboxes.append({
                    "page": page_no,
                    "bbox": bbox_data,
                    "type": type_,
                    "ref": label
                })

        self.e_page = max([bbox["page"] for bbox in chunk_bboxes]) if chunk_bboxes else 0
        self.chunk_bboxes = json.dumps(chunk_bboxes)
        return self

    def set_media_files(self, doc_items: list) -> "GenOSVectorMetaBuilder":
        """청크 내에 포함된 이미지(Picture) 항목들의 정보를 추출하여 저장합니다."""
        temp_list = []
        for item in doc_items:
            if isinstance(item, PictureItem) and item.image:
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({"name": name, "type": "image", "ref": item.self_ref})
        
        self.media_files = json.dumps(temp_list)
        return self

    def build(self) -> GenOSVectorMeta:
        """최종적으로 완성된 GenOSVectorMeta 객체를 생성하여 반환합니다."""
        return GenOSVectorMeta(
            text=self.text,
            n_char=self.n_char,
            n_word=self.n_word,
            n_line=self.n_line,
            i_page=self.i_page,
            e_page=self.e_page,
            i_chunk_on_page=self.i_chunk_on_page,
            n_chunk_of_page=self.n_chunk_of_page,
            i_chunk_on_doc=self.i_chunk_on_doc,
            n_chunk_of_doc=self.n_chunk_of_doc,
            n_page=self.n_page,
            reg_date=self.reg_date,
            chunk_bboxes=self.chunk_bboxes,
            media_files=self.media_files,
            title=self.title,
            created_date=self.created_date,
            appendix=self.appendix or "",
            file_path=self.file_path,
        )

from __future__ import annotations
import json
import os
import logging
import math, bisect
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional, Iterable, Any, List, Dict, Tuple
from fastapi import Request

_log = logging.getLogger(__name__)
import shutil
import subprocess
import tempfile
import unicodedata


def convert_to_pdf(file_path: str, use_pdf_sdk: bool = True) -> str | None:
    from genon.preprocessor.converters.hwp_to_pdf import convert_hwp_to_pdf

    primary = "pdf_sdk" if use_pdf_sdk else "libreoffice"
    return convert_hwp_to_pdf(file_path, primary=primary, disable_fallback=True)


def _is_pdf(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


from docling_core.transforms.chunker import (
    BaseChunk,
    BaseChunker,
    DocChunk,
    DocMeta,
)
from docling_core.types import DoclingDocument
from pandas import DataFrame
import asyncio
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.document import (
    DocumentOrigin,
    LevelNumber,
    ListItem,
    CodeItem,
    ContentLayer,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    DocItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    PageItem,
    ProvenanceItem,
)
from docling.datamodel.settings import settings
from collections import Counter
import re
import json
import warnings
from typing import Iterable, Iterator, Optional, Union
from pydantic import BaseModel, ConfigDict, PositiveInt, TypeAdapter, model_validator
from typing_extensions import Self

try:
    import semchunk
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError:
    raise RuntimeError(
        "Module requires 'chunking' extra; to install, run: "
        "`pip install 'docling-core[chunking]'`"
    )
try:
    from genos_utils import upload_files
except ImportError:
    upload_files = None
from .utils import convert_to_pdf, _is_pdf, assert_cancelled

"""Chunker implementation leveraging the document structure."""


class GenosBucketChunker(BaseChunker):
    """
    문서의 구조(Heading, Table, Picture)를 파악하고, 토큰 제한(max_tokens) 내에서
    표와 캡션이 끊기지 않도록 정밀하게 문서를 분할(Chunking)하는 클래스입니다.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tokenizer: Union[PreTrainedTokenizerBase, str, Path] = (
        Path("/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2")
        if Path(
            "/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2"
        ).exists()
        else "sentence-transformers/all-MiniLM-L6-v2"
    )
    max_tokens: int = 1024
    merge_peers: bool = True
    _tokenizer: PreTrainedTokenizerBase = None
    merge_list_items: bool = True

    @model_validator(mode="after")
    def _initialize_components(self) -> Self:
        self._tokenizer = (
            self.tokenizer
            if isinstance(self.tokenizer, PreTrainedTokenizerBase)
            else AutoTokenizer.from_pretrained(self.tokenizer)
        )
        return self

    def preprocess(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        """
        Docling 문서 객체(dl_doc)를 순회하며 헤딩(Heading) 정보와 리스트 아이템 등을
        1차적으로 병합하고 전처리하여 단일 DocChunk 형태로 반환합니다.
        """
        all_items = []
        all_header_info = []
        current_heading_by_level: dict[LevelNumber, str] = {}
        all_header_short_info = []
        current_heading_short_by_level: dict[LevelNumber, str] = {}
        list_items: list[TextItem] = []
        processed_refs = set()
        for item, level in dl_doc.iterate_items(
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
        ):
            if hasattr(item, "self_ref"):
                processed_refs.add(item.self_ref)
            if not isinstance(item, DocItem):
                continue
            if self.merge_list_items:
                if isinstance(item, ListItem) or (
                    isinstance(item, TextItem) and item.label == DocItemLabel.LIST_ITEM
                ):
                    list_items.append(item)
                    continue
                elif list_items:
                    for list_item in list_items:
                        all_items.append(list_item)
                        all_header_info.append(
                            {k: v for k, v in current_heading_by_level.items()}
                        )
                        all_header_short_info.append(
                            {k: v for k, v in current_heading_short_by_level.items()}
                        )
                    list_items = []
            if isinstance(item, SectionHeaderItem) or (
                isinstance(item, TextItem)
                and item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]
            ):
                header_level = (
                    item.level
                    if isinstance(item, SectionHeaderItem)
                    else (0 if item.label == DocItemLabel.TITLE else 1)
                )
                current_heading_by_level[header_level] = item.text
                current_heading_short_by_level[header_level] = item.orig
                keys_to_del = [k for k in current_heading_by_level if k > header_level]
                for k in keys_to_del:
                    current_heading_by_level.pop(k, None)
                keys_to_del_short = [
                    k for k in current_heading_short_by_level if k > header_level
                ]
                for k in keys_to_del_short:
                    current_heading_short_by_level.pop(k, None)
                all_items.append(item)
                all_header_info.append(
                    {k: v for k, v in current_heading_by_level.items()}
                )
                all_header_short_info.append(
                    {k: v for k, v in current_heading_short_by_level.items()}
                )
                continue
            if (
                isinstance(item, TextItem)
                or isinstance(item, ListItem)
                or isinstance(item, CodeItem)
                or isinstance(item, TableItem)
                or isinstance(item, PictureItem)
            ):
                all_items.append(item)
                all_header_info.append(
                    {k: v for k, v in current_heading_by_level.items()}
                )
                all_header_short_info.append(
                    {k: v for k, v in current_heading_short_by_level.items()}
                )
        if list_items:
            for list_item in list_items:
                all_items.append(list_item)
                all_header_info.append(
                    {k: v for k, v in current_heading_by_level.items()}
                )
                all_header_short_info.append(
                    {k: v for k, v in current_heading_short_by_level.items()}
                )
        missing_tables = []
        for table in dl_doc.tables:
            table_ref = getattr(table, "self_ref", None)
            if table_ref not in processed_refs:
                missing_tables.append(table)
        if missing_tables:
            for missing_table in missing_tables:
                all_items.insert(0, missing_table)
                all_header_info.insert(0, {})
                all_header_short_info.insert(0, {})
        if not all_items:
            return
        chunk = DocChunk(
            text="",
            meta=DocMeta(
                doc_items=all_items,
                headings=None,
                captions=None,
                origin=dl_doc.origin,
            ),
        )
        chunk._header_info_list = all_header_info
        chunk._header_short_info_list = all_header_short_info
        yield chunk

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        max_chunk_length = 300
        total_tokens = 0
        lines = text.split("\n")
        current_chunk = ""
        for line in lines:
            temp_chunk = current_chunk + "\n" + line if current_chunk else line
            if len(temp_chunk) <= max_chunk_length:
                current_chunk = temp_chunk
            else:
                if current_chunk:
                    try:
                        total_tokens += len(self._tokenizer.tokenize(current_chunk))
                    except Exception:
                        total_tokens += int(len(current_chunk.split()) * 1.3)
                current_chunk = line
        if current_chunk:
            try:
                total_tokens += len(self._tokenizer.tokenize(current_chunk))
            except Exception:
                total_tokens += int(len(current_chunk.split()) * 1.3)
        return total_tokens

    def _generate_text_from_items_with_headers(
        self,
        items: list[DocItem],
        header_info_list: list[dict],
        dl_doc: DoclingDocument,
        **kwargs,
    ) -> str:
        text_parts = []
        current_section_headers = {}
        for i, item in enumerate(items):
            item_headers = header_info_list[i] if i < len(header_info_list) else {}
            if item_headers != current_section_headers:
                headers_to_add = []
                for level in sorted(item_headers.keys()):
                    if (
                        level not in current_section_headers
                        or current_section_headers[level] != item_headers[level]
                    ):
                        for l in sorted(item_headers.keys()):
                            if l < level:
                                headers_to_add.append(item_headers[l])
                            elif l == level:
                                headers_to_add.append("")
                        break
                if headers_to_add:
                    header_text = ", ".join(headers_to_add)
                    if header_text not in text_parts:
                        text_parts.append(header_text)
                current_section_headers = item_headers.copy()
            if isinstance(item, TableItem):
                table_text = self._extract_table_text(item, dl_doc, **kwargs)
                if table_text:
                    text_parts.append(table_text)
            elif hasattr(item, "text") and item.text:
                if item.text not in text_parts:
                    text_parts.append(item.text)
            elif isinstance(item, PictureItem):
                text_parts.append("")
        result_text = self.delim.join(text_parts)
        return result_text

    def _extract_table_text(
        self, table_item: TableItem, dl_doc: DoclingDocument, **kwargs
    ) -> str:
        try:
            export_to_html = kwargs.get("export_to_html", 1)
            if export_to_html == 1:
                table_text = table_item.export_to_html(dl_doc)
            else:
                table_text = table_item.export_to_markdown(dl_doc)
            if table_text and table_text.strip():
                return table_text
        except Exception:
            pass
        try:
            if hasattr(table_item, "data") and table_item.data:
                cell_texts = []
                if hasattr(table_item.data, "table_cells"):
                    for cell in table_item.data.table_cells:
                        if hasattr(cell, "text") and cell.text and cell.text.strip():
                            cell_texts.append(cell.text.strip())
                elif hasattr(table_item.data, "grid") and table_item.data.grid:
                    for row in table_item.data.grid:
                        if isinstance(row, list):
                            for cell in row:
                                if (
                                    hasattr(cell, "text")
                                    and cell.text
                                    and cell.text.strip()
                                ):
                                    cell_texts.append(cell.text.strip())
                if cell_texts:
                    return " ".join(cell_texts)
        except Exception:
            pass
        if hasattr(table_item, "text") and table_item.text:
            return table_item.text
        return ""

    def _extract_used_headers(
        self, header_info_list: list[dict]
    ) -> Optional[list[str]]:
        if not header_info_list:
            return None
        all_headers = []
        seen_headers = set()
        for header_info in header_info_list:
            if header_info:
                for level in sorted(header_info.keys()):
                    header_text = header_info[level]
                    if header_text and header_text not in seen_headers:
                        all_headers.append(header_text)
                        seen_headers.add(header_text)
        return all_headers if all_headers else None

    def _split_table_text(self, table_text: str, max_tokens: int) -> list[str]:
        if not table_text:
            return [table_text]
        if self._count_tokens(table_text) <= max_tokens:
            return [table_text]
        chunker = semchunk.chunkerify(self._tokenizer, chunk_size=max_tokens)
        chunks = chunker(table_text)
        return chunks if chunks else [table_text]

    def _is_section_header(self, item: DocItem) -> bool:
        return isinstance(item, SectionHeaderItem) or (
            isinstance(item, TextItem)
            and item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]
        )

    def _get_section_header_level(self, item: DocItem) -> Optional[int]:
        if isinstance(item, SectionHeaderItem):
            return item.level
        elif isinstance(item, TextItem):
            if item.label == DocItemLabel.TITLE:
                return 0
            elif item.label == DocItemLabel.SECTION_HEADER:
                return 1
        return None

    def _generate_section_text_with_heading(
        self,
        section_items: list[DocItem],
        section_header_infos: list[dict],
        dl_doc: DoclingDocument,
        **kwargs,
    ) -> str:
        if section_header_infos and section_header_infos[0]:
            merged_headers = {}
            for level, header_text in section_header_infos[0].items():
                if header_text:
                    merged_headers[level] = header_text
            if merged_headers:
                sorted_levels = sorted(merged_headers.keys())
                headers = [merged_headers[level] for level in sorted_levels]
                heading_text = ", ".join(headers)
            else:
                heading_text = ""
        else:
            heading_text = ""
        section_text = self._generate_text_from_items_with_headers(
            section_items, section_header_infos, dl_doc, **kwargs
        )
        if heading_text:
            return heading_text + ", " + section_text
        else:
            return section_text

    def _split_document_by_tokens(
        self, doc_chunk: DocChunk, dl_doc: DoclingDocument, **kwargs
    ) -> list[DocChunk]:
        """
        단일로 뭉쳐진 전처리된 DocChunk를 받아, 
        1) 섹션 헤더(Heading)를 기준으로 분할하고
        2) 캡션과 표가 쪼개지지 않도록 바운딩 박스를 계산하여
        3) 최대 허용 토큰 수(max_tokens) 내에서 균등하게 청크를 생성합니다.
        """
        items = doc_chunk.meta.doc_items
        header_info_list = getattr(doc_chunk, "_header_info_list", [])
        header_short_info_list = getattr(doc_chunk, "_header_short_info_list", [])
        if not items:
            return []

        def get_header_level(header_infos, *, first=False, default=-1):
            if not header_infos:
                return default
            info = header_infos[0] if first else header_infos[-1]
            return max(info.keys(), default=default)

        def get_current_chunk(
            doc_chunk: DocChunk,
            merged_texts: list[str],
            merged_header_short_infos: list[dict],
            merged_items: list[DocItem],
        ):
            if not merged_texts:
                return None
            chunk_text = "\n".join(merged_texts)
            used_headers = self._extract_used_headers(merged_header_short_infos)
            return DocChunk(
                text=chunk_text,
                meta=DocMeta(
                    doc_items=merged_items,
                    headings=used_headers,
                    captions=None,
                    origin=doc_chunk.meta.origin,
                ),
            )

        def get_text_from_item(item: DocItem) -> str:
            if isinstance(item, TableItem):
                return self._extract_table_text(item, dl_doc, **kwargs)
            elif hasattr(item, "text") and item.text:
                return item.text
            elif isinstance(item, PictureItem):
                text = ""
                for annotation in item.annotations:
                    if hasattr(annotation, "text"):
                        text += annotation.text
                return text
            return ""

        def split_items_evenly_by_tokens(item_token_counts, max_tokens):
            """
            토큰 수 리스트를 받아, 전체 합이 max_tokens를 초과할 경우
            bisect를 활용해 최대한 균등하게 n개의 그룹으로 나눌 분기점(cuts)을 반환합니다.
            """
            n = len(item_token_counts)
            total = sum(item_token_counts)
            if n == 0:
                return []
            if total <= max_tokens:
                return [(0, n)]
            k = math.ceil(total / max_tokens)
            target = total / k
            P = [0]
            for c in item_token_counts:
                P.append(P[-1] + c)
            cuts = [0]
            used = {0}
            for t in range(1, k):
                goal = t * target
                j = bisect.bisect_left(P, goal)
                cand = []
                if 0 < j < len(P):
                    cand.append(j)
                if 0 <= j - 1 < len(P):
                    cand.append(j - 1)
                best = None
                best_dist = float("inf")
                for x in cand:
                    if x in used:
                        continue
                    if x <= cuts[-1]:
                        continue
                    if x >= len(P) - 1:
                        continue
                    dist = abs(P[x] - goal)
                    if dist < best_dist:
                        best_dist = dist
                        best = x
                if best is None:
                    best = min(max(cuts[-1] + 1, 1), len(P) - 2)
                cuts.append(best)
                used.add(best)
            cuts.append(n)
            return [(a, b) for a, b in zip(cuts[:-1], cuts[1:])]

        def adjust_captions(items_group):
            """
            청크가 토큰 한계로 분할될 때, 그림/표의 캡션이 부모 아이템과 
            서로 다른 청크로 찢어지는 현상을 방지하기 위해 강제로 병합합니다.
            """
            b_modified = False
            for idx, group in enumerate(items_group):
                if group is None:
                    continue
                item = group[0][0]
                ref_idx_list = []
                if hasattr(item, "captions") and item.captions:
                    for cap in item.captions:
                        cap_ref = cap.cref
                        cap_idx = -1
                        for j, it in enumerate(items_group):
                            if it is None:
                                continue
                            if getattr(it[0][0], "self_ref", None) == cap_ref:
                                cap_idx = j
                                break
                        if cap_idx != -1:
                            ref_idx_list.append(cap_idx)
                if ref_idx_list:
                    ref_idx_list = sorted(ref_idx_list)
                if not ref_idx_list:
                    continue
                for cap_idx in ref_idx_list:
                    for g in items_group[cap_idx]:
                        items_group[idx].append(g)
                    items_group[cap_idx] = None
                    b_modified = True
            if b_modified:
                items_group = [it for it in items_group if it is not None]
            return items_group

        def adjust_pictures_in_tables(items_group):
            """
            청크가 토큰 한계로 분할될 때, 표(Table) 내부 영역(Bounding Box)에
            포함된 그림(Picture)이 분리되지 않도록 교차 면적(IoS)을 계산하여 병합합니다.
            """
            b_modified = False
            for idx, group in enumerate(items_group):
                if group is None:
                    continue
                item = group[0][0]
                pic_idx_list = []
                if isinstance(item, TableItem):
                    table_bbox = item.prov[0].bbox
                    table_page_no = item.prov[0].page_no
                    for j in range(len(items_group)):
                        if items_group[j] is None:
                            continue
                        pic_item = items_group[j][0][0]
                        if isinstance(pic_item, PictureItem):
                            pic_bbox = pic_item.prov[0].bbox
                            pic_page_no = pic_item.prov[0].page_no
                            if pic_page_no != table_page_no:
                                continue
                            ios = pic_bbox.intersection_over_self(table_bbox)
                            if ios > 0.5:
                                pic_idx_list.append(j)
                    if pic_idx_list:
                        pic_idx_list = sorted(pic_idx_list)
                if not pic_idx_list:
                    continue
                for pic_idx in pic_idx_list:
                    for g in items_group[pic_idx]:
                        items_group[idx].append(g)
                    items_group[pic_idx] = None
                    b_modified = True
            if b_modified:
                items_group = [it for it in items_group if it is not None]
            return items_group

        sections = []
        cur_items, cur_h_infos, cur_h_short = [], [], []
        for i, item in enumerate(items):
            h_info = header_info_list[i] if i < len(header_info_list) else {}
            h_short = (
                header_short_info_list[i] if i < len(header_short_info_list) else {}
            )
            if self._is_section_header(item):
                if cur_items:
                    sections.append((cur_items, cur_h_infos, cur_h_short))
                cur_items = [item]
                cur_h_infos = [h_info]
                cur_h_short = [h_short]
            else:
                cur_items.append(item)
                cur_h_infos.append(h_info)
                cur_h_short.append(h_short)
        if cur_items:
            sections.append((cur_items, cur_h_infos, cur_h_short))
        sections_with_text = []
        for items, header_infos, header_short_infos in sections:
            text = self._generate_section_text_with_heading(
                items, header_short_infos, dl_doc, **kwargs
            )
            sections_with_text.append((text, items, header_infos, header_short_infos))
        if self.max_tokens > 0:
            for i in range(len(sections_with_text)):
                text, items, h_infos, h_short = sections_with_text[i]
                token_count = self._count_tokens(text)
                if token_count < self.max_tokens:
                    continue
                items_group = [
                    [(item, info, short)]
                    for item, info, short in zip(items, h_infos, h_short)
                ]
                items_group = adjust_captions(items_group)
                items_group = adjust_pictures_in_tables(items_group)
                item_token_counts = []
                for group in items_group:
                    cur_count = 0
                    for g in group:
                        cur_count += self._count_tokens(get_text_from_item(g[0]))
                    item_token_counts.append(cur_count)
                split_info = split_items_evenly_by_tokens(
                    item_token_counts, self.max_tokens
                )
                new_sections = []
                for a, b in split_info:
                    group_items = []
                    group_h_infos = []
                    group_h_short = []
                    for idx in range(a, b):
                        for g in items_group[idx]:
                            group_items.append(g[0])
                            group_h_infos.append(g[1])
                            group_h_short.append(g[2])
                    new_text = self._generate_section_text_with_heading(
                        group_items, group_h_short, dl_doc, **kwargs
                    )
                    new_sections.append(
                        (new_text, group_items, group_h_infos, group_h_short)
                    )
                sections_with_text.pop(i)
                for new_section in reversed(new_sections):
                    sections_with_text.insert(i, new_section)
        for i in range(len(sections_with_text) - 2, -1, -1):
            text, items, h_infos, h_short = sections_with_text[i]
            if len(items) != 1 or not self._is_section_header(items[0]):
                continue
            item_text = "".join(getattr(it, "text", "") for it in items)
            if len(item_text) > 30:
                continue
            n_text, n_items, n_h_infos, n_h_short = sections_with_text[i + 1]
            current_level = get_header_level(h_infos, first=False)
            next_level = get_header_level(n_h_infos, first=True)
            if 0 <= next_level < current_level:
                continue
            sections_with_text[i] = (
                text + "\n" + n_text,
                items + n_items,
                h_infos + n_h_infos,
                h_short + n_h_short,
            )
            sections_with_text.pop(i + 1)
        result_chunks = []
        merged_texts, merged_items = [], []
        merged_header_infos, merged_header_short_infos = [], []
        for text, items, header_infos, header_short_infos in sections_with_text:
            b_new_chunk = False
            test_tokens = self._count_tokens("\n".join(merged_texts + [text]))
            section_level = get_header_level(header_infos, first=True)
            merged_level = get_header_level(merged_header_infos, first=False)
            if test_tokens > self.max_tokens and len(merged_texts) > 0:
                b_new_chunk = True
            elif 0 <= section_level < merged_level:
                b_new_chunk = True
            if b_new_chunk:
                cur_chunk = get_current_chunk(
                    doc_chunk, merged_texts, merged_header_short_infos, merged_items
                )
                if cur_chunk:
                    result_chunks.append(cur_chunk)
                merged_texts = [text]
                merged_items = items
                merged_header_infos = header_infos
                merged_header_short_infos = header_short_infos
            else:
                merged_texts.append(text)
                merged_items.extend(items)
                merged_header_infos.extend(header_infos)
                merged_header_short_infos.extend(header_short_infos)
        cur_chunk = get_current_chunk(
            doc_chunk, merged_texts, merged_header_short_infos, merged_items
        )
        if cur_chunk:
            result_chunks.append(cur_chunk)
        return result_chunks

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        """
        문서(DoclingDocument)를 입력받아 전처리를 수행한 뒤,
        토큰 수에 맞게 정밀 분할된 청크(Chunk)들을 순차적으로 반환하는 최종 파이프라인 메서드입니다.
        """
        doc_chunks = list(self.preprocess(dl_doc=dl_doc, **kwargs))
        if not doc_chunks:
            return iter([])
        doc_chunk = doc_chunks[0]
        final_chunks = self._split_document_by_tokens(doc_chunk, dl_doc, **kwargs)
        return iter(final_chunks)

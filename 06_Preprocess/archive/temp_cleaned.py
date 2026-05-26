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
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    LayoutModelType,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureModelType,
    PipelineOptions,
    PaddleOcrOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    FormatOption
)
from docling.datamodel.pipeline_options import DataEnrichmentOptions
from docling.utils.document_enrichment import enrich_document, check_document
from docling.datamodel.document import ConversionResult
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
    ProvenanceItem
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
"""Chunker implementation leveraging the document structure."""
class GenosBucketChunker(BaseChunker):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tokenizer: Union[PreTrainedTokenizerBase, str, Path] = (
            Path("/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2")
            if Path("/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2").exists()
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
        all_items = []
        all_header_info = []  
        current_heading_by_level: dict[LevelNumber, str] = {}
        all_header_short_info = []  
        current_heading_short_by_level: dict[LevelNumber, str] = {}
        list_items: list[TextItem] = []
        processed_refs = set()
        for item, level in dl_doc.iterate_items(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}):
            if hasattr(item, 'self_ref'):
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
                        all_header_info.append({k: v for k, v in current_heading_by_level.items()})
                        all_header_short_info.append({k: v for k, v in current_heading_short_by_level.items()})
                    list_items = []
            if isinstance(item, SectionHeaderItem) or (
                isinstance(item, TextItem) and
                item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]
            ):
                header_level = (
                    item.level if isinstance(item, SectionHeaderItem)
                    else (0 if item.label == DocItemLabel.TITLE else 1)
                )
                current_heading_by_level[header_level] = item.text
                current_heading_short_by_level[header_level] = item.orig  
                keys_to_del = [k for k in current_heading_by_level if k > header_level]
                for k in keys_to_del:
                    current_heading_by_level.pop(k, None)
                keys_to_del_short = [k for k in current_heading_short_by_level if k > header_level]
                for k in keys_to_del_short:
                    current_heading_short_by_level.pop(k, None)
                all_items.append(item)
                all_header_info.append({k: v for k, v in current_heading_by_level.items()})
                all_header_short_info.append({k: v for k, v in current_heading_short_by_level.items()})
                continue
            if (isinstance(item, TextItem) or
                isinstance(item, ListItem) or
                isinstance(item, CodeItem) or
                isinstance(item, TableItem) or
                isinstance(item, PictureItem)):
                all_items.append(item)
                all_header_info.append({k: v for k, v in current_heading_by_level.items()})
                all_header_short_info.append({k: v for k, v in current_heading_short_by_level.items()})
        if list_items:
            for list_item in list_items:
                all_items.append(list_item)
                all_header_info.append({k: v for k, v in current_heading_by_level.items()})
                all_header_short_info.append({k: v for k, v in current_heading_short_by_level.items()})
        missing_tables = []
        for table in dl_doc.tables:
            table_ref = getattr(table, 'self_ref', None)
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
        lines = text.split('\n')
        current_chunk = ""
        for line in lines:
            temp_chunk = current_chunk + '\n' + line if current_chunk else line
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
    def _generate_text_from_items_with_headers(self, items: list[DocItem],
                                              header_info_list: list[dict],
                                              dl_doc: DoclingDocument,
                                              **kwargs) -> str:
        text_parts = []
        current_section_headers = {}  
        for i, item in enumerate(items):
            item_headers = header_info_list[i] if i < len(header_info_list) else {}
            if item_headers != current_section_headers:
                headers_to_add = []
                for level in sorted(item_headers.keys()):
                    if (level not in current_section_headers or
                        current_section_headers[level] != item_headers[level]):
                        for l in sorted(item_headers.keys()):
                            if l < level:
                                headers_to_add.append(item_headers[l])
                            elif l == level:
                                headers_to_add.append('')
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
            elif hasattr(item, 'text') and item.text:
                if item.text not in text_parts:
                    text_parts.append(item.text)
            elif isinstance(item, PictureItem):
                text_parts.append("")  
        result_text = self.delim.join(text_parts)
        return result_text
    def _extract_table_text(self, table_item: TableItem, dl_doc: DoclingDocument, **kwargs) -> str:
        try:
            export_to_html = kwargs.get('export_to_html', 1)
            if export_to_html == 1:
                table_text = table_item.export_to_html(dl_doc)
            else:
                table_text = table_item.export_to_markdown(dl_doc)
            if table_text and table_text.strip():
                return table_text
        except Exception:
            pass
        try:
            if hasattr(table_item, 'data') and table_item.data:
                cell_texts = []
                if hasattr(table_item.data, 'table_cells'):
                    for cell in table_item.data.table_cells:
                        if hasattr(cell, 'text') and cell.text and cell.text.strip():
                            cell_texts.append(cell.text.strip())
                elif hasattr(table_item.data, 'grid') and table_item.data.grid:
                    for row in table_item.data.grid:
                        if isinstance(row, list):
                            for cell in row:
                                if hasattr(cell, 'text') and cell.text and cell.text.strip():
                                    cell_texts.append(cell.text.strip())
                if cell_texts:
                    return ' '.join(cell_texts)
        except Exception:
            pass
        if hasattr(table_item, 'text') and table_item.text:
            return table_item.text
        return ""
    def _extract_used_headers(self, header_info_list: list[dict]) -> Optional[list[str]]:
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
        return (isinstance(item, SectionHeaderItem) or
                (isinstance(item, TextItem) and
                 item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]))
    def _get_section_header_level(self, item: DocItem) -> Optional[int]:
        if isinstance(item, SectionHeaderItem):
            return item.level
        elif isinstance(item, TextItem):
            if item.label == DocItemLabel.TITLE:
                return 0
            elif item.label == DocItemLabel.SECTION_HEADER:
                return 1
        return None
    def _generate_section_text_with_heading(self, section_items: list[DocItem],
                                            section_header_infos: list[dict],
                                            dl_doc: DoclingDocument,
                                            **kwargs) -> str:
        if section_header_infos and section_header_infos[0]:
            merged_headers = {}
            for level, header_text in section_header_infos[0].items():
                if header_text:
                    merged_headers[level] = header_text
            if merged_headers:
                sorted_levels = sorted(merged_headers.keys())
                headers = [merged_headers[level] for level in sorted_levels]
                heading_text = ', '.join(headers)
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
    def _split_document_by_tokens(self, doc_chunk: DocChunk, dl_doc: DoclingDocument, **kwargs) -> list[DocChunk]:
        items = doc_chunk.meta.doc_items
        header_info_list = getattr(doc_chunk, '_header_info_list', [])
        header_short_info_list = getattr(doc_chunk, '_header_short_info_list', [])
        if not items:
            return []
        def get_header_level(header_infos, *, first=False, default=-1):
            if not header_infos:
                return default
            info = header_infos[0] if first else header_infos[-1]
            return max(info.keys(), default=default)
        def get_current_chunk(doc_chunk: DocChunk, merged_texts: list[str], merged_header_short_infos: list[dict], merged_items: list[DocItem]):
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
                    )
                )
        def get_text_from_item(item: DocItem) -> str:
            if isinstance(item, TableItem):
                return self._extract_table_text(item, dl_doc, **kwargs)
            elif hasattr(item, 'text') and item.text:
                return item.text
            elif isinstance(item, PictureItem):
                text = ""
                for annotation in item.annotations:
                    if hasattr(annotation, 'text'):
                        text += annotation.text
                return text
            return ""
        def split_items_evenly_by_tokens(item_token_counts, max_tokens):
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
                if 0 < j < len(P): cand.append(j)
                if 0 <= j-1 < len(P): cand.append(j-1)
                best = None
                best_dist = float("inf")
                for x in cand:
                    if x in used:
                        continue
                    if x <= cuts[-1]:
                        continue
                    if x >= len(P)-1:  
                        continue
                    dist = abs(P[x] - goal)
                    if dist < best_dist:
                        best_dist = dist
                        best = x
                if best is None:
                    best = min(max(cuts[-1] + 1, 1), len(P)-2)
                cuts.append(best)
                used.add(best)
            cuts.append(n)
            return [(a, b) for a, b in zip(cuts[:-1], cuts[1:])]
        def adjust_captions(items_group):
            b_modified = False
            for idx, group in enumerate(items_group):
                if group is None:
                    continue
                item = group[0][0]
                ref_idx_list = []
                if hasattr(item, 'captions') and item.captions:
                    for cap in item.captions:
                        cap_ref = cap.cref
                        cap_idx = -1
                        for j, it in enumerate(items_group):
                            if it is None:
                                continue
                            if getattr(it[0][0], 'self_ref', None) == cap_ref:
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
            h_short = header_short_info_list[i] if i < len(header_short_info_list) else {}
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
            sections_with_text.append((
                text,
                items,
                header_infos,
                header_short_infos
            ))
        if self.max_tokens > 0:
            for i in range(len(sections_with_text)):
                text, items, h_infos, h_short = sections_with_text[i]
                token_count = self._count_tokens(text)
                if token_count < self.max_tokens:
                    continue
                items_group=[[(item, info, short)] for item, info, short in zip(items, h_infos, h_short)]
                items_group = adjust_captions(items_group)
                items_group = adjust_pictures_in_tables(items_group)
                item_token_counts = []
                for group in items_group:
                    cur_count = 0
                    for g in group:
                        cur_count += self._count_tokens(get_text_from_item(g[0]))
                    item_token_counts.append(cur_count)
                split_info = split_items_evenly_by_tokens(item_token_counts, self.max_tokens)
                new_sections = []
                for (a, b) in split_info:
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
                    new_sections.append((new_text, group_items, group_h_infos, group_h_short))
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
            sections_with_text[i] = (text + '\n' + n_text, items + n_items, h_infos + n_h_infos, h_short + n_h_short)
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
                cur_chunk = get_current_chunk(doc_chunk, merged_texts, merged_header_short_infos, merged_items)
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
        cur_chunk = get_current_chunk(doc_chunk, merged_texts, merged_header_short_infos, merged_items)
        if cur_chunk:
            result_chunks.append(cur_chunk)
        return result_chunks
    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        doc_chunks = list(self.preprocess(dl_doc=dl_doc, **kwargs))
        if not doc_chunks:
            return iter([])
        doc_chunk = doc_chunks[0]  
        final_chunks = self._split_document_by_tokens(doc_chunk, dl_doc, **kwargs)
        return iter(final_chunks)
class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow'
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
        self.text = text
        self.n_char = len(text)
        self.n_word = len(text.split())
        self.n_line = len(text.splitlines())
        return self
    def set_page_info(
            self, i_page: int, i_chunk_on_page: int, n_chunk_of_page: int
    ) -> "GenOSVectorMetaBuilder":
        self.i_page = i_page
        self.i_chunk_on_page = i_chunk_on_page
        self.n_chunk_of_page = n_chunk_of_page
        return self
    def set_chunk_index(self, i_chunk_on_doc: int) -> "GenOSVectorMetaBuilder":
        self.i_chunk_on_doc = i_chunk_on_doc
        return self
    def set_global_metadata(self, **global_metadata) -> "GenOSVectorMetaBuilder":
        for key, value in global_metadata.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    def set_chunk_bboxes(self, doc_items: list, document: DoclingDocument) -> "GenOSVectorMetaBuilder":
        chunk_bboxes = []
        for item in doc_items:
            for prov in item.prov:
                label = item.self_ref
                type_ = item.label
                size = document.pages.get(prov.page_no).size
                page_no = prov.page_no
                bbox = prov.bbox
                bbox_data = {'l': bbox.l / size.width,
                             't': bbox.t / size.height,
                             'r': bbox.r / size.width,
                             'b': bbox.b / size.height,
                             'coord_origin': bbox.coord_origin.value}
                chunk_bboxes.append({'page': page_no, 'bbox': bbox_data, 'type': type_, 'ref': label})
        self.e_page = max([bbox['page'] for bbox in chunk_bboxes]) if chunk_bboxes else 0
        self.chunk_bboxes = json.dumps(chunk_bboxes)
        return self
    def set_media_files(self, doc_items: list) -> "GenOSVectorMetaBuilder":
        temp_list = []
        for item in doc_items:
            if isinstance(item, PictureItem) and item.image:
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'name': name, 'type': 'image', 'ref': item.self_ref})
        self.media_files = json.dumps(temp_list)
        return self
    def build(self) -> GenOSVectorMeta:
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
class DocumentProcessor:
    def __init__(self):
        self.ocr_endpoint = "http://192.168.73.172:48080/ocr"
        ocr_options = PaddleOcrOptions(
            force_full_page_ocr=False,
            lang=['korean'],
            ocr_endpoint=self.ocr_endpoint,
            text_score=0.3)
        self.page_chunk_counts = defaultdict(int)
        device = AcceleratorDevice.AUTO
        num_threads = 8
        accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)
        self.pipe_line_options = PdfPipelineOptions()
        self.pipe_line_options.generate_page_images = True
        self.pipe_line_options.generate_picture_images = True
        self.pipe_line_options.do_ocr = False
        self.pipe_line_options.ocr_options = ocr_options
        self.pipe_line_options.images_scale = 2
        self.pipe_line_options.layout_options.layout_model_type = LayoutModelType.GENOS_LAYOUT
        self.pipe_line_options.layout_options.genos_layout_options.endpoint = "http://192.168.75.174:26001/v1/chat/completions"
        self.pipe_line_options.layout_options.genos_layout_options.api_key = ""
        settings.perf.page_batch_size = 32
        self.pipe_line_options.do_table_structure = True
        self.pipe_line_options.table_structure_options.do_cell_matching = True
        self.pipe_line_options.table_structure_options.mode = TableFormerMode.ACCURATE
        self.pipe_line_options.accelerator_options = accelerator_options
        self.simple_pipeline_options = PipelineOptions()
        self.simple_pipeline_options.save_images = False
        self.ocr_pipe_line_options = PdfPipelineOptions()
        self.ocr_pipe_line_options = self.pipe_line_options.model_copy(deep=True)
        self.ocr_pipe_line_options.do_ocr = True
        self.ocr_pipe_line_options.ocr_options = ocr_options.model_copy(deep=True)
        self.ocr_pipe_line_options.ocr_options.force_full_page_ocr = True
        self._create_converters()
        self.enrichment_options = DataEnrichmentOptions(
            do_toc_enrichment=True,
            toc_doc_type="law",
            extract_metadata=True,
            toc_api_provider="custom",
            toc_api_base_url="https://genos.genon.ai:3443/api/gateway/rep/serving/502/v1/chat/completions",
            metadata_api_base_url="https://genos.genon.ai:3443/api/gateway/rep/serving/502/v1/chat/completions",
            toc_api_key="022653a3743849e299f19f19d323490b",
            metadata_api_key="022653a3743849e299f19f19d323490b",
            toc_model="model",
            metadata_model="model",
            toc_temperature=0.0,
            toc_top_p=0.00001,
            toc_seed=33,
            toc_max_tokens=10000,
            toc_system_prompt=toc_system_prompt,
            toc_user_prompt=toc_user_prompt,
        )
    def _create_converters(self):
        self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=self.pipe_line_options,
                        backend=PyPdfiumDocumentBackend
                    ),
                }
            )
        self.second_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipe_line_options,
                    backend=PyPdfiumDocumentBackend
                ),
            },
        )
        self.ocr_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=self.ocr_pipe_line_options,
                        backend=DoclingParseV4DocumentBackend
                    ),
                }
            )
        self.ocr_second_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.ocr_pipe_line_options,
                    backend=PyPdfiumDocumentBackend
                ),
            },
        )
    def load_documents_with_docling(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        save_images = kwargs.get('save_images', True)
        include_wmf = kwargs.get('include_wmf', False)
        if (self.simple_pipeline_options.save_images != save_images or
            getattr(self.simple_pipeline_options, 'include_wmf', False) != include_wmf):
            self.simple_pipeline_options.save_images = save_images
            self.simple_pipeline_options.include_wmf = include_wmf
            self._create_converters()
        try:
            conv_result: ConversionResult = self.converter.convert(file_path, raises_on_error=True)
        except Exception as e:
            conv_result: ConversionResult = self.second_converter.convert(file_path, raises_on_error=True)
        return conv_result.document
    def load_documents_with_docling_ocr(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        save_images = kwargs.get('save_images', True)
        include_wmf = kwargs.get('include_wmf', False)
        if (self.simple_pipeline_options.save_images != save_images or
            getattr(self.simple_pipeline_options, 'include_wmf', False) != include_wmf):
            self.simple_pipeline_options.save_images = save_images
            self.simple_pipeline_options.include_wmf = include_wmf
            self._create_converters()
        try:
            conv_result: ConversionResult = self.ocr_converter.convert(file_path, raises_on_error=True)
        except Exception as e:
            conv_result: ConversionResult = self.ocr_second_converter.convert(file_path, raises_on_error=True)
        return conv_result.document
    def load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        return self.load_documents_with_docling(file_path, **kwargs)
    def split_documents(self, documents: DoclingDocument, **kwargs: dict) -> List[DocChunk]:
        chunker: GenosBucketChunker = GenosBucketChunker(
            max_tokens = kwargs.get('max_chunk_size', 0),
            merge_peers = True
        )
        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=documents, **kwargs))
        for chunk in chunks:
            if chunk.meta.doc_items[0].prov:
                self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks
    def safe_join(self, iterable):
        if not isinstance(iterable, (list, tuple, set)):
            return ''
        return ''.join(map(str, iterable)) + '\n'
    def parse_created_date(self, date_text: str) -> Optional[int]:
        if not date_text or not isinstance(date_text, str) or date_text == "None":
            return 0
        date_text = date_text.strip()
        match_full = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', date_text)
        if match_full:
            year, month, day = match_full.groups()
            try:
                datetime(int(year), int(month), int(day))
                return int(f"{year}{month.zfill(2)}{day.zfill(2)}")
            except ValueError:
                pass
        match_month = re.match(r'^(\d{4})-(\d{1,2})$', date_text)
        if match_month:
            year, month = match_month.groups()
            try:
                datetime(int(year), int(month), 1)
                return int(f"{year}{month.zfill(2)}01")
            except ValueError:
                pass
        match_year = re.match(r'^(\d{4})$', date_text)
        if match_year:
            year = match_year.group(1)
            try:
                datetime(int(year), 1, 1)
                return int(f"{year}0101")
            except ValueError:
                pass
        return 0
    def enrichment(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        document = enrich_document(document, self.enrichment_options, **kwargs)
        return document
    async def compose_vectors(self, document: DoclingDocument, chunks: List[DocChunk], file_path: str, request: Request, converted_pdf_path: Optional[str] = None, **kwargs: dict) ->            list[dict]:
        title = ""
        created_date = 0
        try:
            if (document.key_value_items and
                len(document.key_value_items) > 0 and
                hasattr(document.key_value_items[0], 'graph') and
                hasattr(document.key_value_items[0].graph, 'cells') and
                len(document.key_value_items[0].graph.cells) > 1):
                date_text = document.key_value_items[0].graph.cells[1].text
                created_date = self.parse_created_date(date_text)
        except (AttributeError, IndexError) as e:
            pass
        for item, _ in document.iterate_items():
            if hasattr(item, 'label'):
                if item.label == DocItemLabel.TITLE:
                    title = item.text.strip() if item.text else ""
                    break
        appendix_info = kwargs.get('appendix', '')
        appendix_list = []
        if isinstance(appendix_info, str):
            appendix_list = [item.strip() for item in json.loads(appendix_info) if item.strip()] if appendix_info else []
        elif isinstance(appendix_info, list):
            appendix_list = appendix_info
        else:
            appendix_list = []
        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=document.num_pages(),
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z',
            created_date=created_date,
            title=title
        )
        if converted_pdf_path:
            global_metadata['file_path'] = converted_pdf_path
        current_page = None
        chunk_index_on_page = 0
        vectors = []
        upload_tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_page = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items[0].prov else 0
            headers_text = "HEADER: " + ", ".join(chunk.meta.headings) + '\n' if chunk.meta.headings else ''
            content = headers_text + chunk.text
            matched_appendices = self.check_appendix_keywords(content, appendix_list)
            chunk_global_metadata = global_metadata.copy()
            chunk_global_metadata['appendix'] = matched_appendices  
            if chunk_page != current_page:
                current_page = chunk_page
                chunk_index_on_page = 0
            vector = (GenOSVectorMetaBuilder()
                      .set_text(content)
                      .set_page_info(chunk_page, chunk_index_on_page, self.page_chunk_counts[chunk_page])
                      .set_chunk_index(chunk_idx)
                      .set_global_metadata(**chunk_global_metadata) 
                      .set_chunk_bboxes(chunk.meta.doc_items, document)
                      .set_media_files(chunk.meta.doc_items)
                      ).build()
            vectors.append(vector)
            chunk_index_on_page += 1
            if upload_files:
                file_list = self.get_media_files(chunk.meta.doc_items)
                upload_tasks.append(asyncio.create_task(
                    upload_files(file_list, request=request)
                ))
        if upload_tasks:
            await asyncio.gather(*upload_tasks)
        return vectors
    def get_media_files(self, doc_items: list):
        temp_list = []
        for item in doc_items:
            if isinstance(item, PictureItem):
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'path': path, 'name': name})
        return temp_list
    def check_glyph_text(self, text: str, threshold: int = 1) -> bool:
        if not text:
            return False
        matches = re.findall(r'GLYPH\w*', text)
        if len(matches) >= threshold:
            return True
        return False
    def check_glyphs(self, document: DoclingDocument) -> bool:
        for item, level in document.iterate_items():
            if isinstance(item, TextItem) and hasattr(item, 'prov') and item.prov:
                page_no = item.prov[0].page_no
                matches = re.findall(r'GLYPH\w*', item.text)
                if len(matches) > 10:
                    return True
        return False
    def check_appendix_keywords(self, content: str, appendix_list: list) -> str: 
        if not content or not appendix_list:
            return ""
        matched_appendices = []
        found_patterns = []
        content = re.sub(r"\s+", "", content)
        complex_patterns = re.findall(r'(별지|별표|장부)(?:제)?([^<>()\[\]]+?)(?=(?:호|서식)|[<>\)\]]|$)', content)
        for pattern_type, number in complex_patterns:
            found_patterns.extend([
                f"{pattern_type} {number}",
                f"{pattern_type} 제{number}호",
                f"{pattern_type}{number}",
                f"{pattern_type}제{number}호"
            ])
        standalone_patterns = re.findall(r'[\(\[]+(별지|별표|장부)[\)\]]+', content)
        for pattern_type in set(standalone_patterns):
            found_patterns.extend([
                pattern_type,
                f"{pattern_type}",
            ])
        for appendix in appendix_list:
            if not appendix or not isinstance(appendix, str):
                continue
            appendix_clean = appendix.replace('.pdf', '').lower().strip()
            for pattern in found_patterns:
                if pattern.lower().strip() in appendix_clean:
                    matched_appendices.append(appendix)
                    break  
        return ', '.join(matched_appendices) if matched_appendices else ""
    def ocr_all_table_cells(self, document: DoclingDocument, pdf_path) -> List[Dict[str, Any]]:
        import fitz
        import base64
        import requests
        def post_ocr_bytes(img_bytes: bytes, timeout=60) -> dict:
            HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}
            payload = {"file": base64.b64encode(img_bytes).decode("ascii"), "fileType": 1, "visualize": False}
            r = requests.post(self.ocr_endpoint, json=payload, headers=HEADERS, timeout=timeout)
            if not r.ok:
                raise RuntimeError(f"OCR HTTP {r.status_code}: {r.text[:500]}")
            return r.json()
        def extract_ocr_fields(resp: dict):
            if resp is None:
                return [], [], []
            if resp.get("errorCode") not in (0, None):
                return [], [], []
            ocr_results = (
                resp.get("result", {})
                    .get("ocrResults", [])
            )
            if not ocr_results:
                return [], [], []
            pruned = (
                ocr_results[0]
                .get("prunedResult", {})
            )
            if not pruned:
                return [], [], []
            rec_texts  = pruned.get("rec_texts", [])   
            rec_scores = pruned.get("rec_scores", [])  
            rec_boxes  = pruned.get("rec_boxes", [])   
            n = min(len(rec_texts), len(rec_scores), len(rec_boxes))
            return rec_texts[:n], rec_scores[:n], rec_boxes[:n]
        try:
            doc = fitz.open(pdf_path)
            for table_idx, table_item in enumerate(document.tables):
                if not table_item.data or not table_item.data.table_cells:
                    continue
                b_ocr = False
                for cell_idx, cell in enumerate(table_item.data.table_cells):
                    if self.check_glyph_text(cell.text, threshold=1):
                        b_ocr = True
                        break
                if b_ocr is False:
                    continue
                for cell_idx, cell in enumerate(table_item.data.table_cells):
                    if not table_item.prov:
                        continue
                    page_no = table_item.prov[0].page_no - 1
                    bbox = cell.bbox
                    page = doc.load_page(page_no)
                    cell_bbox = fitz.Rect(
                        bbox.l, min(bbox.t, bbox.b),
                        bbox.r, max(bbox.t, bbox.b)
                    )
                    bbox_height = cell_bbox.height
                    target_height = 20
                    zoom_factor = target_height / bbox_height if bbox_height > 0 else 1.0
                    zoom_factor = min(zoom_factor, 4.0)  
                    zoom_factor = max(zoom_factor, 1)  
                    mat = fitz.Matrix(zoom_factor, zoom_factor)
                    pix = page.get_pixmap(matrix=mat, clip=cell_bbox)
                    img_data = pix.tobytes("png")
                    result = post_ocr_bytes(img_data, timeout=60)
                    rec_texts, rec_scores, rec_boxes = extract_ocr_fields(result)
                    cell.text = ""
                    for t in rec_texts:
                        if len(cell.text) > 0:
                            cell.text += " "
                        cell.text += t if t else ""
        except Exception as e:
            print(f"OCR processing failed: {e}")
            pass
        return document
    def setup_logging(self, level_num: int):
        def get_level_name(level_num: int) -> str:
            level_map = {
                5: "DEBUG",
                4: "INFO",
                3: "WARNING",
                2: "ERROR",
                1: "CRITICAL",
                0: "NOLOG"
            }
            return level_map.get(level_num, "INFO")
        level_name = get_level_name(level_num)
        print(f"Setting log level to: {level_name}")
        if level_name == "NOLOG" or not hasattr(logging, level_name):
            logging.disable(logging.CRITICAL)  
            return
        level = getattr(logging, level_name.upper())
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler()]   
        )
        logging.getLogger().setLevel(level)
    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        self.setup_logging(kwargs.get('log_level', 4))
        _log.info(f"file_path: {file_path}")
        _log.info(f"kwargs: {kwargs}")
        converted_pdf_path: Optional[str] = None
        if kwargs.get('auto_convert_to_pdf', True) and not _is_pdf(file_path):
            _log.info(f"[intelligent] Non-PDF input — auto-converting to PDF: {file_path}")
            use_sdk = kwargs.get('use_pdf_sdk', True)
            converted = convert_to_pdf(file_path, use_pdf_sdk=use_sdk)
            if (not converted or not os.path.exists(converted)) and use_sdk:
                _log.warning(f"[intelligent] SDK conversion failed → fallback to LibreOffice")
                converted = convert_to_pdf(file_path, use_pdf_sdk=False)
            if not converted or not os.path.exists(converted):
                raise GenosServiceException(1, f"PDF 변환 실패: {file_path}")
            file_path = converted
            converted_pdf_path = converted
            _log.info(f"[intelligent] Converted PDF: {file_path}")
        document: DoclingDocument = self.load_documents(file_path, **kwargs)
        if not check_document(document, self.enrichment_options) or self.check_glyphs(document):
            document: DoclingDocument = self.load_documents_with_docling_ocr(file_path, **kwargs)
        document: DoclingDocument = self.ocr_all_table_cells(document, file_path)
        output_path, output_file = os.path.split(file_path)
        filename, _ = os.path.splitext(output_file)
        artifacts_dir = Path(f"{output_path}/{filename}")
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = artifacts_dir.parent
        document = document._with_pictures_refs(image_dir=artifacts_dir, page_no=None, reference_path=reference_path)
        document = self.enrichment(document, **kwargs)
        has_text_items = False
        for item, _ in document.iterate_items():
            if (isinstance(item, (TextItem, ListItem, CodeItem, SectionHeaderItem)) and item.text and item.text.strip()) or (isinstance(item, TableItem) and item.data and len(item.data.table_cells) == 0):
                has_text_items = True
                break
        if has_text_items:
            chunks: List[DocChunk] = self.split_documents(document, **kwargs)
        else:
            page_no = 1
            prov = ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(l=0, t=0, r=1, b=1),  
                charspan=(0, 1)
            )
            document.add_text(
                label=DocItemLabel.TEXT,
                text=".",
                prov=prov
            )
            chunks: List[DocChunk] = self.split_documents(document, **kwargs)
        vectors = []
        if len(chunks) >= 1:
            vectors: list[dict] = await self.compose_vectors(
                document, chunks, file_path, request,
                converted_pdf_path=converted_pdf_path,
                **kwargs,
            )
        else:
            raise GenosServiceException(1, f"chunk length is 0")
        if converted_pdf_path and upload_files:
            original_name = kwargs.get('file_name') or os.path.basename(converted_pdf_path)
            pdf_object_name = os.path.splitext(original_name)[0] + '.pdf'
            await upload_files(
                [{'path': converted_pdf_path, 'name': pdf_object_name}],
                request=request,
            )
        """
        # 미디어 파일 업로드 방법
        media_files = [
            { 'path': '/tmp/graph.jpg', 'name': 'graph.jpg', 'type': 'image' },
            { 'path': '/result/1/graph.jpg', 'name': '1/graph.jpg', 'type': 'image' },
        ]
        # 업로드 요청 시에는 path, name 필요
        file_list = [{k: v for k, v in file.items() if k != 'type'} for file in media_files]
        await upload_files(file_list, request=request)
        # 메타에 저장시에는 name, type 필요
        meta = [{k: v for k, v in file.items() if k != 'path'} for file in media_files]
        vectors[0].media_files = meta
        """
        return vectors
class GenosServiceException(Exception):
    def __init__(self, error_code: str, error_msg: Optional[str] = None, msg_params: Optional[dict] = None) -> None:
        self.code = 1
        self.error_code = error_code
        self.error_msg = error_msg or "GenOS Service Exception"
        self.msg_params = msg_params or {}
    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(code={self.code!r}, errMsg={self.error_msg!r})"
async def assert_cancelled(request: Request):
    if await request.is_disconnected():
        raise GenosServiceException(1, f"Cancelled")
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
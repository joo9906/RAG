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

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
    PipelineOptions,
    EasyOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption, FormatOption
from pydantic import BaseModel
class DataEnrichmentOptions(BaseModel):
    do_toc_enrichment: bool = False
    toc_doc_type: str = ""
    extract_metadata: bool = False
    toc_api_provider: str = ""
    toc_api_base_url: str = ""
    metadata_api_base_url: str = ""
    toc_api_key: str = ""
    metadata_api_key: str = ""
    toc_model: str = ""
    metadata_model: str = ""
    toc_temperature: float = 0.0
    toc_top_p: float = 0.0
    toc_seed: int = 0
    toc_max_tokens: int = 0
    toc_system_prompt: str = ""
    toc_user_prompt: str = ""

def enrich_document(document, options, **kwargs):
    return document

def check_document(document, options):
    return True

from docling.datamodel.document import ConversionResult
from docling_core.transforms.chunker import (
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
from ip_modules.utils import (
    convert_to_pdf,
    _is_pdf,
    assert_cancelled,
    GenosServiceException,
    toc_system_prompt,
    toc_user_prompt,
)
from ip_modules.meta_builder import GenOSVectorMetaBuilder, GenOSVectorMeta
from ip_modules.chunker import GenosBucketChunker


class DocumentProcessor:
    """
    Docling을 기반으로 문서를 파싱하고, OCR 및 전처리를 수행하며,
    최종적으로 Chunk 배열과 JSON 벡터 데이터를 조립하는 메인 파이프라인(Orchestrator) 클래스입니다.
    """
    def __init__(self):
        self.ocr_endpoint = "http://192.168.73.172:48080/ocr"
        ocr_options = EasyOcrOptions(
            force_full_page_ocr=False,
            lang=["ko"],
        )
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
        # self.pipe_line_options.layout_options.layout_model_type = (
        #     LayoutModelType.GENOS_LAYOUT
        # )
        # self.pipe_line_options.layout_options.genos_layout_options.endpoint = (
        #     "http://192.168.75.174:26001/v1/chat/completions"
        # )
        # self.pipe_line_options.layout_options.genos_layout_options.api_key = ""
        settings.perf.page_batch_size = 32
        self.pipe_line_options.do_table_structure = True
        self.pipe_line_options.table_structure_options.do_cell_matching = True
        self.pipe_line_options.table_structure_options.mode = TableFormerMode.ACCURATE
        self.pipe_line_options.accelerator_options = accelerator_options
        self.simple_pipeline_options = {"save_images": False, "include_wmf": False}
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
                    backend=PyPdfiumDocumentBackend,
                ),
            }
        )
        self.second_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipe_line_options,
                    backend=PyPdfiumDocumentBackend,
                ),
            },
        )
        self.ocr_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.ocr_pipe_line_options,
                    backend=DoclingParseV4DocumentBackend,
                ),
            }
        )
        self.ocr_second_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.ocr_pipe_line_options,
                    backend=PyPdfiumDocumentBackend,
                ),
            },
        )

    def load_documents_with_docling(
        self, file_path: str, **kwargs: dict
    ) -> DoclingDocument:
        save_images = kwargs.get("save_images", True)
        include_wmf = kwargs.get("include_wmf", False)
        if (
            self.simple_pipeline_options.get("save_images", True) != save_images
            or self.simple_pipeline_options.get("include_wmf", False) != include_wmf
        ):
            self.simple_pipeline_options["save_images"] = save_images
            self.simple_pipeline_options["include_wmf"] = include_wmf
            self._create_converters()
        try:
            conv_result: ConversionResult = self.converter.convert(
                file_path, raises_on_error=True
            )
        except Exception as e:
            conv_result: ConversionResult = self.second_converter.convert(
                file_path, raises_on_error=True
            )
        return conv_result.document

    def load_documents_with_docling_ocr(
        self, file_path: str, **kwargs: dict
    ) -> DoclingDocument:
        save_images = kwargs.get("save_images", True)
        include_wmf = kwargs.get("include_wmf", False)
        if (
            self.simple_pipeline_options.get("save_images", True) != save_images
            or self.simple_pipeline_options.get("include_wmf", False) != include_wmf
        ):
            self.simple_pipeline_options["save_images"] = save_images
            self.simple_pipeline_options["include_wmf"] = include_wmf
            self._create_converters()
        try:
            conv_result: ConversionResult = self.ocr_converter.convert(
                file_path, raises_on_error=True
            )
        except Exception as e:
            conv_result: ConversionResult = self.ocr_second_converter.convert(
                file_path, raises_on_error=True
            )
        return conv_result.document

    def load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        return self.load_documents_with_docling(file_path, **kwargs)

    def split_documents(
        self, documents: DoclingDocument, **kwargs: dict
    ) -> List[DocChunk]:
        chunker: GenosBucketChunker = GenosBucketChunker(
            max_tokens=kwargs.get("max_chunk_size", 0), merge_peers=True
        )
        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=documents, **kwargs))
        for chunk in chunks:
            if chunk.meta.doc_items[0].prov:
                self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks

    def safe_join(self, iterable):
        if not isinstance(iterable, (list, tuple, set)):
            return ""
        return "".join(map(str, iterable)) + "\n"

    def parse_created_date(self, date_text: str) -> Optional[int]:
        if not date_text or not isinstance(date_text, str) or date_text == "None":
            return 0
        date_text = date_text.strip()
        match_full = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", date_text)
        if match_full:
            year, month, day = match_full.groups()
            try:
                datetime(int(year), int(month), int(day))
                return int(f"{year}{month.zfill(2)}{day.zfill(2)}")
            except ValueError:
                pass
        match_month = re.match(r"^(\d{4})-(\d{1,2})$", date_text)
        if match_month:
            year, month = match_month.groups()
            try:
                datetime(int(year), int(month), 1)
                return int(f"{year}{month.zfill(2)}01")
            except ValueError:
                pass
        match_year = re.match(r"^(\d{4})$", date_text)
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

    async def compose_vectors(
        self,
        document: DoclingDocument,
        chunks: List[DocChunk],
        file_path: str,
        request: Request,
        converted_pdf_path: Optional[str] = None,
        **kwargs: dict,
    ) -> list[dict]:
        title = ""
        created_date = 0
        try:
            if (
                document.key_value_items
                and len(document.key_value_items) > 0
                and hasattr(document.key_value_items[0], "graph")
                and hasattr(document.key_value_items[0].graph, "cells")
                and len(document.key_value_items[0].graph.cells) > 1
            ):
                date_text = document.key_value_items[0].graph.cells[1].text
                created_date = self.parse_created_date(date_text)
        except (AttributeError, IndexError) as e:
            pass
        for item, _ in document.iterate_items():
            if hasattr(item, "label"):
                if item.label == DocItemLabel.TITLE:
                    title = item.text.strip() if item.text else ""
                    break
        appendix_info = kwargs.get("appendix", "")
        appendix_list = []
        if isinstance(appendix_info, str):
            appendix_list = (
                [item.strip() for item in json.loads(appendix_info) if item.strip()]
                if appendix_info
                else []
            )
        elif isinstance(appendix_info, list):
            appendix_list = appendix_info
        else:
            appendix_list = []
        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=document.num_pages(),
            reg_date=datetime.now().isoformat(timespec="seconds") + "Z",
            created_date=created_date,
            title=title,
        )
        if converted_pdf_path:
            global_metadata["file_path"] = converted_pdf_path
        current_page = None
        chunk_index_on_page = 0
        vectors = []
        upload_tasks = []

        # ==============================================================================
        # Option 2 (Batch): OpenAI Batch API를 이용한 50% 토큰 절약 요약 (비동기 대기 방식)
        # ==============================================================================
        inject_summary = kwargs.get("inject_summary", False)
        batch_id = None

        if inject_summary == "batch":
            from ip_modules.batch_utils import create_and_submit_batch
            
            # 전체 문서 텍스트 추출 (앞 5만 자)
            document_content = "".join([
                item.text + "\n" for item, _ in document.iterate_items() 
                if hasattr(item, "text") and item.text
            ])
            if len(document_content) > 50000:
                document_content = document_content[:50000] + "...(생략)"
                
            # 청크 텍스트 리스트 준비
            chunks_text = []
            for chunk in chunks:
                c_headers = ("HEADER: " + ", ".join(chunk.meta.headings) + "\n" if chunk.meta.headings else "")
                chunks_text.append(c_headers + chunk.text)
                
            _log.info(f"OpenAI Batch API로 {len(chunks)}개 청크 요약본 생성 요청을 전송합니다...")
            batch_id = await create_and_submit_batch(
                chunks_text=chunks_text, 
                document_content=document_content,
                model_name="gpt-4o-mini"
            )
            if batch_id:
                _log.info(f"Batch 요청 성공! 나중에 배치 ID({batch_id})로 결과를 확인하세요.")

        # 청크 순회 및 최종 JSON(Vector) 조립
        for chunk_idx, chunk in enumerate(chunks):
            chunk_page = (
                chunk.meta.doc_items[0].prov[0].page_no
                if chunk.meta.doc_items[0].prov
                else 0
            )
            headers_text = (
                "HEADER: " + ", ".join(chunk.meta.headings) + "\n"
                if chunk.meta.headings
                else ""
            )
            content = headers_text + chunk.text
            
            matched_appendices = self.check_appendix_keywords(content, appendix_list)
            chunk_global_metadata = global_metadata.copy()
            chunk_global_metadata["appendix"] = matched_appendices
            
            # Batch ID가 발급되었다면 메타데이터에 임시 기록해둡니다.
            if batch_id:
                chunk_global_metadata["pending_batch_id"] = batch_id
            if chunk_page != current_page:
                current_page = chunk_page
                chunk_index_on_page = 0
            vector = (
                GenOSVectorMetaBuilder()
                .set_text(content)
                .set_page_info(
                    chunk_page, chunk_index_on_page, self.page_chunk_counts[chunk_page]
                )
                .set_chunk_index(chunk_idx)
                .set_global_metadata(**chunk_global_metadata)
                .set_chunk_bboxes(chunk.meta.doc_items, document)
                .set_media_files(chunk.meta.doc_items)
            ).build()
            vectors.append(vector)
            chunk_index_on_page += 1
            if upload_files:
                file_list = self.get_media_files(chunk.meta.doc_items)
                upload_tasks.append(
                    asyncio.create_task(upload_files(file_list, request=request))
                )
        if upload_tasks:
            await asyncio.gather(*upload_tasks)
        return vectors

    def get_media_files(self, doc_items: list):
        temp_list = []
        for item in doc_items:
            if isinstance(item, PictureItem):
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({"path": path, "name": name})
        return temp_list

    def check_glyph_text(self, text: str, threshold: int = 1) -> bool:
        if not text:
            return False
        matches = re.findall(r"GLYPH\w*", text)
        if len(matches) >= threshold:
            return True
        return False

    def check_glyphs(self, document: DoclingDocument) -> bool:
        for item, level in document.iterate_items():
            if isinstance(item, TextItem) and hasattr(item, "prov") and item.prov:
                page_no = item.prov[0].page_no
                matches = re.findall(r"GLYPH\w*", item.text)
                if len(matches) > 10:
                    return True
        return False

    def check_appendix_keywords(self, content: str, appendix_list: list) -> str:
        if not content or not appendix_list:
            return ""
        matched_appendices = []
        found_patterns = []
        content = re.sub(r"\s+", "", content)
        complex_patterns = re.findall(
            r"(별지|별표|장부)(?:제)?([^<>()\[\]]+?)(?=(?:호|서식)|[<>\)\]]|$)", content
        )
        for pattern_type, number in complex_patterns:
            found_patterns.extend(
                [
                    f"{pattern_type} {number}",
                    f"{pattern_type} 제{number}호",
                    f"{pattern_type}{number}",
                    f"{pattern_type}제{number}호",
                ]
            )
        standalone_patterns = re.findall(r"[\(\[]+(별지|별표|장부)[\)\]]+", content)
        for pattern_type in set(standalone_patterns):
            found_patterns.extend(
                [
                    pattern_type,
                    f"{pattern_type}",
                ]
            )
        for appendix in appendix_list:
            if not appendix or not isinstance(appendix, str):
                continue
            appendix_clean = appendix.replace(".pdf", "").lower().strip()
            for pattern in found_patterns:
                if pattern.lower().strip() in appendix_clean:
                    matched_appendices.append(appendix)
                    break
        return ", ".join(matched_appendices) if matched_appendices else ""

    def ocr_all_table_cells(
        self, document: DoclingDocument, pdf_path
    ) -> List[Dict[str, Any]]:
        """
        문서 내의 표(Table) 중에서 GLYPH 문자가 포함된 깨진 셀을 찾아내고,
        해당 위치의 PDF 영역(Bounding Box) 이미지를 캡처하여
        별도의 OCR API 엔드포인트에 요청 후 정확한 텍스트로 보정합니다.
        """
        import fitz
        import base64
        import requests

        def post_ocr_bytes(img_bytes: bytes, timeout=60) -> dict:
            HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}
            payload = {
                "file": base64.b64encode(img_bytes).decode("ascii"),
                "fileType": 1,
                "visualize": False,
            }
            r = requests.post(
                self.ocr_endpoint, json=payload, headers=HEADERS, timeout=timeout
            )
            if not r.ok:
                raise RuntimeError(f"OCR HTTP {r.status_code}: {r.text[:500]}")
            return r.json()

        def extract_ocr_fields(resp: dict):
            if resp is None:
                return [], [], []
            if resp.get("errorCode") not in (0, None):
                return [], [], []
            ocr_results = resp.get("result", {}).get("ocrResults", [])
            if not ocr_results:
                return [], [], []
            pruned = ocr_results[0].get("prunedResult", {})
            if not pruned:
                return [], [], []
            rec_texts = pruned.get("rec_texts", [])
            rec_scores = pruned.get("rec_scores", [])
            rec_boxes = pruned.get("rec_boxes", [])
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
                        bbox.l, min(bbox.t, bbox.b), bbox.r, max(bbox.t, bbox.b)
                    )
                    bbox_height = cell_bbox.height
                    target_height = 20
                    zoom_factor = (
                        target_height / bbox_height if bbox_height > 0 else 1.0
                    )
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
                0: "NOLOG",
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
            handlers=[logging.StreamHandler()],
        )
        logging.getLogger().setLevel(level)

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        """
        사용자 요청이 들어왔을 때 실행되는 메인 진입점(Entry Point)입니다.
        
        1) PDF가 아닐 경우 변환(convert_to_pdf)
        2) Docling 파이프라인 수행 (기본 파싱 -> 실패 시 OCR 강제 파싱)
        3) 표 이미지 캡처 후 부분 OCR 보정 (ocr_all_table_cells)
        4) 토큰 단위/표 단위 최적화 청킹 (GenosBucketChunker)
        5) 부록 매칭 및 최종 JSON Vector 데이터 조립 (compose_vectors)
        """
        self.setup_logging(kwargs.get("log_level", 4))
        _log.info(f"file_path: {file_path}")
        _log.info(f"kwargs: {kwargs}")
        converted_pdf_path: Optional[str] = None
        if kwargs.get("auto_convert_to_pdf", True) and not _is_pdf(file_path):
            _log.info(
                f"[intelligent] Non-PDF input — auto-converting to PDF: {file_path}"
            )
            use_sdk = kwargs.get("use_pdf_sdk", True)
            converted = convert_to_pdf(file_path, use_pdf_sdk=use_sdk)
            if (not converted or not os.path.exists(converted)) and use_sdk:
                _log.warning(
                    f"[intelligent] SDK conversion failed → fallback to LibreOffice"
                )
                converted = convert_to_pdf(file_path, use_pdf_sdk=False)
            if not converted or not os.path.exists(converted):
                raise GenosServiceException(1, f"PDF 변환 실패: {file_path}")
            file_path = converted
            converted_pdf_path = converted
            _log.info(f"[intelligent] Converted PDF: {file_path}")
        document: DoclingDocument = self.load_documents(file_path, **kwargs)
        if not check_document(document, self.enrichment_options) or self.check_glyphs(
            document
        ):
            document: DoclingDocument = self.load_documents_with_docling_ocr(
                file_path, **kwargs
            )
        document: DoclingDocument = self.ocr_all_table_cells(document, file_path)
        output_path, output_file = os.path.split(file_path)
        filename, _ = os.path.splitext(output_file)
        artifacts_dir = Path(f"{output_path}/{filename}")
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = artifacts_dir.parent
        document = document._with_pictures_refs(
            image_dir=artifacts_dir, page_no=None, reference_path=reference_path
        )
        document = self.enrichment(document, **kwargs)
        has_text_items = False
        for item, _ in document.iterate_items():
            if (
                isinstance(item, (TextItem, ListItem, CodeItem, SectionHeaderItem))
                and item.text
                and item.text.strip()
            ) or (
                isinstance(item, TableItem)
                and item.data
                and len(item.data.table_cells) == 0
            ):
                has_text_items = True
                break
        if has_text_items:
            chunks: List[DocChunk] = self.split_documents(document, **kwargs)
        else:
            page_no = 1
            prov = ProvenanceItem(
                page_no=page_no, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1)
            )
            document.add_text(label=DocItemLabel.TEXT, text=".", prov=prov)
            chunks: List[DocChunk] = self.split_documents(document, **kwargs)
        vectors = []
        if len(chunks) >= 1:
            vectors: list[dict] = await self.compose_vectors(
                document,
                chunks,
                file_path,
                request,
                converted_pdf_path=converted_pdf_path,
                **kwargs,
            )
        else:
            raise GenosServiceException(1, f"chunk length is 0")
        if converted_pdf_path and upload_files:
            original_name = kwargs.get("file_name") or os.path.basename(
                converted_pdf_path
            )
            pdf_object_name = os.path.splitext(original_name)[0] + ".pdf"
            await upload_files(
                [{"path": converted_pdf_path, "name": pdf_object_name}],
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

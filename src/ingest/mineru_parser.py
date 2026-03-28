import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import MarkdownTextSplitter
from src.utils.cache import get_tokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from mineru.cli.common import do_parse, read_fn

    MINERU_AVAILABLE = True
except ImportError:
    do_parse = None  # type: ignore[assignment]
    read_fn = None  # type: ignore[assignment]
    MINERU_AVAILABLE = False
    logger.warning(
        "mineru package not fully installed or configured. Using mock parser for skeleton execution."
    )
    logger.warning("Ensure you have Python >= 3.10 and installed 'mineru'.")

# Common ways to write references section titles
_REFERENCE_HEADINGS = frozenset(
    [
        "REFERENCES",
        "REFERENCE",
        "BIBLIOGRAPHY",
        "WORKS CITED",
        "LITERATURE CITED",
        "CITED LITERATURE",
    ]
)

# Common title keywords at the beginning of the text
_MAIN_CONTENT_HEADINGS = frozenset(
    [
        "ABSTRACT",
        "INTRODUCTION",
        "SUMMARY",
        "OVERVIEW",
    ]
)

# Extract the leading chapter number, such as "4", "4.1", "4.1.1"（Allow numbers to be followed directly by text without spaces.）
_SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)")

# Suspected inline subtitle: such as "5.1.3 Compared baselines."
_INLINE_SUBHEADING_RE = re.compile(
    r"^(\d+(\.\d+)+\s*[^.\n?]{1,150}?[.?!])(.*)$", flags=re.DOTALL
)

# Matches only end-of-line hyphen word breakers (hyphen followed by whitespace), used to splice words across lines
_EOL_HYPHEN_RE = re.compile(r"-\s+")

# extract Figure/Table etc. local label
_VISUAL_LABEL_RE = re.compile(
    r"^\s*((?:Figure|Fig\.?|Table|Algorithm)\s+\d+[A-Za-z]?)",
    flags=re.IGNORECASE,
)


def _merge_hyphen_lines(lines: List[str]) -> str:
    """Merge multiple lines of text and fix word breaks caused by hyphens at the end of lines.

    example：["state-of-the-", "art method"] → "state-of-the-art method"
    Note: Only remove the white space after the hyphen at the end of the line, retain the hyphen in the word (such as "self-attention"）。
    """
    result = ""
    for i, line in enumerate(lines):
        if i == 0:
            result = line
        elif result.endswith("-"):
            # The end of the line is a hyphen: direct splicing (remove the white space after the hyphen)）
            result = result + line.lstrip()
        else:
            result = result + " " + line
    return result


class MinerUParser:
    """
    Parser utilizing local MinerU to extract text, figures, and structured data
    from Research PDF papers.
    """

    def __init__(self, output_dir: str = "./output", backend: str = "pipeline"):
        self.output_dir = output_dir
        self.backend = backend
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def backend_subdir(self) -> str:
        """Returns the output subdirectory name corresponding to backend。"""
        if self.backend in ("vlm", "hybrid-auto-engine"):
            return "hybrid_auto"
        return "auto"

    def _scan_output_files(
        self, local_output_dir: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Scan the MinerU output directory and return (middle_json_path, md_path, content_list_path)。"""
        target_json: Optional[str] = None
        target_md: Optional[str] = None
        target_content_list: Optional[str] = None
        for root, _, files in os.walk(local_output_dir):
            for file in files:
                if file.endswith("_middle.json"):
                    target_json = os.path.join(root, file)
                elif file.endswith("_content_list.json") and not file.endswith(
                    "_content_list_v2.json"
                ):
                    target_content_list = os.path.join(root, file)
                elif (
                    file.endswith(".md")
                    and not file.endswith("_clean.md")
                    and not target_md
                ):
                    target_md = os.path.join(root, file)
        return target_json, target_md, target_content_list

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extracts content from a given PDF using MinerU.

        If the output file already exists, read it directly and skip re-parsing (idempotent design）。
        Returns a dictionary with raw markdown, parsed blocks, and metadata.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        # MinerU A folder named pdf_name will be automatically created under the specified output_dir.
        local_output_dir = os.path.join(self.output_dir, pdf_name)

        if MINERU_AVAILABLE:
            try:
                # First check whether the output file already exists to avoid repeated parsing
                target_json, target_md, target_content_list = self._scan_output_files(
                    local_output_dir
                )
                if not target_json:
                    logger.info(
                        "Parsing %s with MinerU (%s backend)...", pdf_name, self.backend
                    )
                    assert read_fn is not None and do_parse is not None

                    # Mirror what the MinerU CLI does before calling do_parse:
                    # set MINERU_MODEL_SOURCE so the VLM/hybrid backend can
                    # locate model weights on ModelScope/HuggingFace.
                    from config.settings import config as _cfg  # noqa: PLC0415

                    if os.getenv("MINERU_MODEL_SOURCE") is None:
                        os.environ["MINERU_MODEL_SOURCE"] = _cfg.MINERU_MODEL_SOURCE

                    pdf_bytes = read_fn(Path(pdf_path))
                    do_parse(
                        output_dir=self.output_dir,
                        pdf_file_names=[pdf_name],
                        pdf_bytes_list=[pdf_bytes],
                        p_lang_list=["en"],
                        backend=self.backend,
                        parse_method="auto",
                        f_dump_md=True,
                        f_dump_orig_pdf=False,
                        f_dump_content_list=True,
                        f_dump_middle_json=True,
                    )
                    target_json, target_md, target_content_list = (
                        self._scan_output_files(local_output_dir)
                    )
                else:
                    logger.info(
                        "Found existing MinerU output for %s, skipping re-parse.",
                        pdf_name,
                    )

                raw_json_data: Dict[str, Any] = {}
                if target_json and os.path.exists(target_json):
                    with open(target_json, "r", encoding="utf-8") as f:
                        raw_json_data = json.load(f)

                content_list_data: List[Dict[str, Any]] = []
                if target_content_list and os.path.exists(target_content_list):
                    with open(target_content_list, "r", encoding="utf-8") as f:
                        content_list_data = json.load(f)

                md_content = ""
                if target_md and os.path.exists(target_md):
                    with open(target_md, "r", encoding="utf-8") as f:
                        md_content = f.read()

                return {
                    "pdf_name": pdf_name,
                    "title": pdf_name,
                    "markdown": md_content,
                    "middle_json": raw_json_data,
                    "content_list": content_list_data,
                }

            except Exception as e:
                logger.error("MinerU parsing failed: %s", e)
                return {
                    "pdf_name": pdf_name,
                    "title": pdf_name,
                    "markdown": f"MinerU parsing failed: {e}",
                    "middle_json": {},
                    "content_list": [],
                }
        else:
            logger.info("Mock analyzing PDF: %s...", pdf_name)
            return {
                "pdf_name": pdf_name,
                "title": pdf_name,
                "markdown": f"# {pdf_name}\n\nMock data.",
                "content_list": [],
                "middle_json": {
                    "pdf_info": [
                        {
                            "page_idx": 0,
                            "para_blocks": [
                                {
                                    "type": "text",
                                    "lines": [
                                        {
                                            "spans": [
                                                {
                                                    "type": "text",
                                                    "content": "Mock data.",
                                                }
                                            ]
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                },
            }

    def chunk_content(
        self, parsed_data: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Parse structured output into retrievable chunks.

        Both pipeline and VLM/hybrid backends are processed through
        ``process_middle_json()``.  The VLM middle.json is structurally
        compatible with the pipeline format; the only differences (``list``
        sub_type, ``code`` blocks, extra ``discarded_blocks`` types, ``angle``
        field) are all handled inside ``process_middle_json()`` already.

        Fallback: if no middle_json is present, split the raw ``*.md`` with
        MarkdownTextSplitter.
        """
        middle_json = parsed_data.get("middle_json", {})
        if middle_json:
            return self.process_middle_json(middle_json)

        md_text = parsed_data.get("markdown", "")
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = splitter.split_text(md_text)
        chunks = [
            {
                "content": chunk,
                "type": "text",
                "metadata": {
                    "chunk_order": index,
                    "page_chunk_order": index,
                    "section_path": "",
                    "section_depth": 0,
                    "local_label": "",
                    "backend": self.backend,
                    "has_caption": False,
                    "has_footnote": False,
                    "has_image": False,
                    "has_equation_images": False,
                    "figure_or_table_label": "",
                },
            }
            for index, chunk in enumerate(text_chunks)
        ]
        return chunks, {}

    def process_middle_json(
        self,
        middle_data: Dict[str, Any],
        max_chunk_size: int = 1500,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Will middle.json Convert to a retrievable list of chunks.

        Fix content：
        - [P0] Eliminate heading breadcrumb and section header double output
        - [P0] Fixed an issue where the breadcrumb of the chart chunk was recorded before the heading was updated.
        - [P0] Fixed References being completely lost (list block was skipped when in_references）
        - [P1] Fix heading_stack level inference: regular expression is changed to extract pure numeric prefix, support no space format
        - [P1] Non-chapter titles (e.g. Algorithm 1:）Do not update heading_stack as local label
        - [P1] list block Correct processing: merge multiple rows into a single entry, and also collect them in references
        - [P2] Remove redundant parts from chart chunks "Figure"/"Table" primitive type tag
        """
        chunks: List[Dict[str, Any]] = []
        doc_metadata: Dict[str, Any] = {
            "pre_abstract_meta": [],
            "footnotes_and_discarded": [],
            "references": [],
            "title_extracted": "",
        }

        # Accumulate the status of text chunks
        current_text_chunk: List[str] = []
        current_chunk_length: int = 0
        current_equation_imgs: List[str] = []
        current_page_idx: int = 0
        current_page_chunk_order: int = 0
        chunk_order: int = 0

        # Multi-level title stack：[(level, heading_text), ...]
        # level Inferred from numeric prefix（"4" → 1, "4.1" → 2, "4.1.1" → 3）
        # Non-numeric titles (e.g. "Algorithm 1:"）level=0，Not pushed into the stack, only local labels
        heading_stack: List[tuple[int, str]] = []

        # The current local label (non-chapter number title), cleared with the next chapter title
        local_label: str = ""

        main_content_started: bool = False
        in_references: bool = False

        try:
            tokenizer = get_tokenizer("cl100k_base")
        except Exception:
            tokenizer = None

        # ------------------------------------------------------------------ #
        # Helper function                                                             #
        # ------------------------------------------------------------------ #

        def count_tokens(text: str) -> int:
            if tokenizer:
                return len(tokenizer.encode(text))
            return len(text) // 4

        def _spans_to_text(spans: List[Dict[str, Any]]) -> str:
            """Splice a list of spans into text and correctly handle inline formulas。"""
            parts: List[str] = []
            for s in spans:
                content = s.get("content", "")
                if s.get("type") == "inline_equation":
                    content = content.strip()
                    if content:
                        content = f" ${content}$ "
                parts.append(content)
            return "".join(parts)

        def get_text(block: Dict[str, Any]) -> str:
            """
            Recursively extract the plain text content of a block.
            processing level：blocks → lines → spans
            """
            text_parts: List[str] = []

            if "blocks" in block:
                for b in block["blocks"]:
                    t = get_text(b)
                    if t:
                        text_parts.append(t)
            elif "lines" in block:
                for line in block["lines"]:
                    line_text = _spans_to_text(line.get("spans", []))
                    if line_text:
                        text_parts.append(line_text)
            elif "spans" in block:
                line_text = _spans_to_text(block["spans"])
                if line_text:
                    text_parts.append(line_text)

            raw_text = "\n".join(text_parts)

            b_type = block.get("type", "")
            if b_type in (
                "text",
                "title",
                "image_caption",
                "table_caption",
                "image_footnote",
                "table_footnote",
            ):
                # Combine hyphens and line breaks (English word breakers）
                raw_text = re.sub(r"-\n\s*", "", raw_text)
                # Merge Chinese line breaks
                raw_text = re.sub(r"([^\x00-\x7F])\n([^\x00-\x7F])", r"\1\2", raw_text)
                raw_text = raw_text.replace("\n", " ")

            return raw_text.strip()

        def get_list_items(block: Dict[str, Any]) -> List[str]:
            """
            Extract the complete text of each list entry from the list block.
            Supports two structures：
            - Pipeline: block include directly lines
            - VLM: block Contains secondary blocks (each block contains lines)
            MinerU's list block uses is_list_start_line to mark the entry start line.
            Multiple lines of the same entry (indented continuation lines) need to be merged。
            """
            items: List[str] = []
            current_item_lines: List[str] = []

            # VLM backend: Secondary blocks structure
            if "blocks" in block:
                for sub_block in block.get("blocks", []):
                    sub_lines = sub_block.get("lines", [])
                    for line in sub_lines:
                        line_text = _spans_to_text(line.get("spans", [])).strip()
                        if not line_text:
                            continue
                        is_start = line.get("is_list_start_line", False)
                        if is_start and current_item_lines:
                            merged = _merge_hyphen_lines(current_item_lines)
                            items.append(merged.strip())
                            current_item_lines = []
                        current_item_lines.append(line_text)
                # last entry
                if current_item_lines:
                    merged = _merge_hyphen_lines(current_item_lines)
                    items.append(merged.strip())
            else:
                # Pipeline backend: direct lines structure
                lines = block.get("lines", [])
                for line in lines:
                    line_text = _spans_to_text(line.get("spans", [])).strip()
                    if not line_text:
                        continue
                    is_start = line.get("is_list_start_line", False)
                    if is_start and current_item_lines:
                        merged = _merge_hyphen_lines(current_item_lines)
                        items.append(merged.strip())
                        current_item_lines = []
                    current_item_lines.append(line_text)

                # last entry
                if current_item_lines:
                    merged = _merge_hyphen_lines(current_item_lines)
                    items.append(merged.strip())

            return [it for it in items if it]

        def get_image_path(block: Dict[str, Any]) -> str:
            """
            Find image paths in each level of the block.
            compatible 'img_path'（MinerU official documentation) and 'image_path'（old field name）。
            """
            for key in ("img_path", "image_path"):
                if key in block:
                    return block[key]
            if "spans" in block:
                for s in block["spans"]:
                    for key in ("img_path", "image_path"):
                        if key in s:
                            return s[key]
            if "lines" in block:
                for line in block["lines"]:
                    for s in line.get("spans", []):
                        for key in ("img_path", "image_path"):
                            if key in s:
                                return s[key]
            if "blocks" in block:
                for b in block["blocks"]:
                    res = get_image_path(b)
                    if res:
                        return res
            return ""

        def get_caption_and_footnote(block: Dict[str, Any]) -> tuple[str, str]:
            """
            from image / table Extract all caption and footnote text in one level block.
            return (caption_text, footnote_text)。
            """
            captions: List[str] = []
            footnotes: List[str] = []
            for sub in block.get("blocks", []):
                sub_type = sub.get("type", "")
                text = get_text(sub)
                if not text:
                    continue
                if "caption" in sub_type:
                    captions.append(text)
                elif "footnote" in sub_type:
                    footnotes.append(text)
            return " ".join(captions), " ".join(footnotes)

        def infer_heading_level(heading_text: str) -> int:
            """
            Infer title level based on numeric prefix（1-based）。
            '1INTRODUCTION'    → 1  (prefix "1"，0 point)
            '2.1 Method'       → 2  (prefix "2.1"，1 point)
            '4.1.1Details'     → 3  (prefix "4.1.1"，2 point)
            No numeric prefix          → 0  (Non-chapter titles, not pushed into the stack)
            """
            m = _SECTION_NUM_RE.match(heading_text.strip())
            if not m:
                return 0  # Non-chapter title
            number_part = m.group(1)
            return number_part.count(".") + 1

        def update_heading_stack(heading_text: str) -> bool:
            """
            Maintain a multi-level title stack, pop out sibling and sub-level titles and push in new titles.
            Return True to indicate that it has been pushed onto the stack (chapter title), False to indicate that it has not been pushed onto the stack (not a chapter title)）。
            """
            nonlocal local_label
            level = infer_heading_level(heading_text)
            if level == 0:
                # Non-chapter headings (e.g. "Algorithm 1:"）：Recorded as a local label and does not modify the stack
                local_label = heading_text.strip()
                return False
            # Chapter title: Clear local labels, update stack
            local_label = ""
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text.strip()))
            return True

        def current_heading_path() -> str:
            """Returns the current title path string, such as '4METHODOLOGY > 4.1 Background'。"""
            path = " > ".join(h for _, h in heading_stack)
            if local_label:
                return f"{path} > {local_label}" if path else local_label
            return path

        def infer_visual_label(text: str) -> str:
            """Extract labels like 'Table 3' or 'Figure 4' from captions."""
            match = _VISUAL_LABEL_RE.match(text.strip())
            return match.group(1) if match else ""

        def append_chunk(
            content: str,
            chunk_type: str,
            metadata: Dict[str, Any],
        ) -> None:
            nonlocal chunk_order, current_page_chunk_order

            section_path = str(metadata.get("heading", current_heading_path()) or "")
            caption = str(metadata.get("caption", "") or "")
            footnote = str(metadata.get("footnote", "") or "")
            img_path = str(metadata.get("img_path", "") or "")

            enriched_metadata = dict(metadata)
            enriched_metadata.setdefault("heading", section_path)
            enriched_metadata.setdefault("section_path", section_path)
            enriched_metadata.setdefault(
                "section_depth", len(heading_stack) + (1 if local_label else 0)
            )
            enriched_metadata.setdefault("local_label", local_label)
            enriched_metadata.setdefault("backend", self.backend)
            enriched_metadata["chunk_order"] = chunk_order
            enriched_metadata["page_chunk_order"] = current_page_chunk_order
            enriched_metadata.setdefault("has_caption", bool(caption))
            enriched_metadata.setdefault("has_footnote", bool(footnote))
            enriched_metadata.setdefault("has_image", bool(img_path))
            enriched_metadata.setdefault("has_equation_images", False)
            enriched_metadata.setdefault(
                "figure_or_table_label", infer_visual_label(caption)
            )

            chunks.append(
                {
                    "content": content,
                    "type": chunk_type,
                    "metadata": enriched_metadata,
                }
            )
            chunk_order += 1
            current_page_chunk_order += 1

        def split_large_text(text: str) -> List[str]:
            """
            Perform tiktoken-aware secondary segmentation on a single piece of text exceeding max_chunk_size。
            """
            if count_tokens(text) <= max_chunk_size:
                return [text]
            sentences = re.split(r"(?<=[.!?])\s+", text)
            sub_chunks: List[str] = []
            buf: List[str] = []
            buf_len = 0
            for sent in sentences:
                sent_len = count_tokens(sent)
                if buf_len + sent_len > max_chunk_size and buf:
                    sub_chunks.append(" ".join(buf))
                    buf, buf_len = [], 0
                buf.append(sent)
                buf_len += sent_len
            if buf:
                sub_chunks.append(" ".join(buf))
            return sub_chunks if sub_chunks else [text]

        def flush_text_chunk() -> None:
            nonlocal current_text_chunk, current_chunk_length, current_equation_imgs
            if not current_text_chunk:
                return
            heading_prefix = current_heading_path()

            # heading_prefix As a prefix, chunk content is not written repeatedly
            prefix = f"[{heading_prefix}]\n\n" if heading_prefix else ""
            combined_text = prefix + "\n\n".join(current_text_chunk)

            meta: Dict[str, Any] = {
                "heading": heading_prefix,
                "page_idx": current_page_idx,
            }
            if current_equation_imgs:
                meta["equation_imgs"] = current_equation_imgs.copy()
                meta["has_equation_images"] = True

            append_chunk(combined_text, "text", meta)
            current_text_chunk = []
            current_chunk_length = 0
            current_equation_imgs = []

        # ------------------------------------------------------------------ #
        # Main traversal logic                                                           #
        # ------------------------------------------------------------------ #

        pdf_info = middle_data.get("pdf_info", [])
        for page_data in pdf_info:
            current_page_idx = page_data.get("page_idx", current_page_idx)
            current_page_chunk_order = 0

            # pipeline Backend: Triggered by keywords; forced to open after page 3
            if current_page_idx >= 3:
                main_content_started = True

            for discard in page_data.get("discarded_blocks", []):
                t = get_text(discard)
                if t:
                    doc_metadata["footnotes_and_discarded"].append(t)

            for block in page_data.get("para_blocks", []):
                b_type = block.get("type", "")

                # -------- title -------- #
                if b_type == "title":
                    text_content = get_text(block)
                    if not text_content:
                        continue

                    # The first title (page 0 or page 1) is considered the paper title
                    if not doc_metadata["title_extracted"]:
                        doc_metadata["title_extracted"] = text_content
                        append_chunk(
                            f"# {text_content}",
                            "title",
                            {
                                "heading": "Title",
                                "section_path": "Title",
                                "section_depth": 1,
                                "page_idx": current_page_idx,
                            },
                        )
                        continue

                    # Detection of main content begins（ABSTRACT / INTRODUCTION and other keywords）
                    if text_content.strip().upper() in _MAIN_CONTENT_HEADINGS:
                        main_content_started = True

                    # Detect reference section
                    if text_content.strip().upper() in _REFERENCE_HEADINGS:
                        in_references = True
                        flush_text_chunk()
                        update_heading_stack(text_content.strip())
                        # References The section title alone serves as the starting point of a text chunk
                        # （current_text_chunk is not pushed, filled by subsequent entries）
                        continue

                    # Ordinary chapter titles (appendices, etc.): exit the reference area and resume text processing
                    # Note: heading itself is not pushed current_text_chunk，
                    # flush_text_chunk() will automatically current_heading_path() Write as prefix
                    if in_references:
                        in_references = False
                    flush_text_chunk()
                    update_heading_stack(text_content.strip())
                    continue

                # -------- references area -------- #
                if in_references:
                    # text and list types are collected as reference entries
                    if b_type == "list":
                        items = get_list_items(block)
                        doc_metadata["references"].extend(items)
                    else:
                        text_content = get_text(block)
                        if text_content:
                            doc_metadata["references"].append(text_content)
                    continue

                # -------- pre-abstract metadata -------- #
                if not main_content_started:
                    text_content = get_text(block)
                    if text_content:
                        doc_metadata["pre_abstract_meta"].append(text_content)
                    continue

                # -------- table -------- #
                if b_type == "table":
                    flush_text_chunk()
                    img_path = get_image_path(block)
                    caption, footnote = get_caption_and_footnote(block)
                    heading_prefix = current_heading_path()
                    # [P2] Remove redundant "Table" primitive type tag
                    parts: List[str] = [f"[{heading_prefix}]"]
                    if caption:
                        parts.append(f"Caption: {caption}")
                    if footnote:
                        parts.append(f"Footnote: {footnote}")
                    if img_path:
                        parts.append(f"Image Path: {img_path}")
                    append_chunk(
                        "\n".join(parts),
                        "table",
                        {
                            "heading": heading_prefix,
                            "page_idx": current_page_idx,
                            "img_path": img_path,
                            "caption": caption,
                            "footnote": footnote,
                        },
                    )

                # -------- image -------- #
                elif b_type == "image":
                    flush_text_chunk()
                    img_path = get_image_path(block)
                    caption, footnote = get_caption_and_footnote(block)
                    heading_prefix = current_heading_path()
                    # [P2] Remove redundant "Figure" primitive type tag
                    parts = [f"[{heading_prefix}]"]
                    if caption:
                        parts.append(f"Caption: {caption}")
                    if footnote:
                        parts.append(f"Footnote: {footnote}")
                    if img_path:
                        parts.append(f"Image Path: {img_path}")
                    append_chunk(
                        "\n".join(parts),
                        "image",
                        {
                            "heading": heading_prefix,
                            "page_idx": current_page_idx,
                            "img_path": img_path,
                            "caption": caption,
                            "footnote": footnote,
                        },
                    )

                # -------- interline equation -------- #
                elif b_type in ("interline_equation", "equation"):
                    text_content = get_text(block)
                    if text_content:
                        current_text_chunk.append(f"\n$$\n{text_content}\n$$\n")
                        current_chunk_length += count_tokens(text_content)
                    eq_img = get_image_path(block)
                    if eq_img:
                        current_equation_imgs.append(eq_img)

                # -------- list block -------- #
                elif b_type == "list":
                    # VLM rear end：sub_type="ref_text" Directly included in the references，
                    # More reliable than relying on REFERENCES title keyword detection。
                    if block.get("sub_type") == "ref_text":
                        ref_items = get_list_items(block)
                        doc_metadata["references"].extend(ref_items)
                        continue

                    # [P1] Correctly merge multiple rows of items using get_list_items
                    items = get_list_items(block)
                    if not items:
                        continue
                    list_text = "\n".join(f"- {item}" for item in items)
                    token_count = count_tokens(list_text)
                    if current_chunk_length + token_count > max_chunk_size:
                        flush_text_chunk()
                    current_text_chunk.append(list_text)
                    current_chunk_length += token_count

                # -------- code / algorithm (VLM backend) -------- #
                elif b_type == "code":
                    flush_text_chunk()
                    sub_type = block.get("sub_type", "code")
                    code_body = ""
                    code_caption = ""
                    for sub in block.get("blocks", []):
                        sub_block_type = sub.get("type", "")
                        if sub_block_type == "code_body":
                            code_body = get_text(sub)
                        elif sub_block_type == "code_caption":
                            code_caption = get_text(sub)
                    heading_prefix = current_heading_path()
                    lang = "text" if sub_type == "algorithm" else "python"
                    parts = [f"[{heading_prefix}]"]
                    if code_caption:
                        parts.append(code_caption)
                    parts.append(f"```{lang}\n{code_body}\n```")
                    append_chunk(
                        "\n".join(parts),
                        "code",
                        {
                            "heading": heading_prefix,
                            "page_idx": current_page_idx,
                            "sub_type": sub_type,
                            "caption": code_caption,
                        },
                    )

                # -------- plain text -------- #
                else:
                    text_content = get_text(block)
                    if not text_content:
                        continue

                    # Detect suspected inline subtitles (e.g. "5.1.3 Compared baselines."）
                    m_sub = _INLINE_SUBHEADING_RE.match(text_content)
                    if m_sub:
                        text_content = f"[{m_sub.group(1)}] {m_sub.group(3).lstrip()}"

                    for part in split_large_text(text_content):
                        part_tokens = count_tokens(part)
                        if current_chunk_length + part_tokens > max_chunk_size:
                            flush_text_chunk()
                        current_text_chunk.append(part)
                        current_chunk_length += part_tokens

        flush_text_chunk()
        return chunks, doc_metadata

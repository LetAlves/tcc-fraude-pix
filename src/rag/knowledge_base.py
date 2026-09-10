"""Ingestão, limpeza e chunking rastreável da base de conhecimento do RAG."""

from __future__ import annotations

import hashlib
import html as html_module
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "rag" / "raw"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "rag" / "manifest.json"
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "rag" / "processed" / "parent_chunks.jsonl"
DEFAULT_PARENT_CHUNK_TOKENS = 500
DEFAULT_PARENT_OVERLAP_TOKENS = 50


class TokenizerProtocol(Protocol):
    """Parte mínima da interface de tokenizador usada pelo pipeline."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str: ...


def encode_without_truncation(tokenizer: TokenizerProtocol, text: str) -> list[int]:
    """Tokeniza todo o texto e desativa o aviso enganoso de limite do modelo."""

    try:
        return tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
    except TypeError:
        return tokenizer.encode(text, add_special_tokens=False)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Manifesto inválido: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_raw_path(raw_dir: Path, filename: object) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"Nome de arquivo inseguro no manifesto: {filename!r}")
    raw_dir = raw_dir.resolve()
    path = (raw_dir / filename).resolve()
    if path.parent != raw_dir:
        raise ValueError(f"Arquivo fora do diretório do corpus: {filename!r}")
    return path


def _normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _base_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "document_id",
        "title",
        "issuer",
        "document_type",
        "source_url",
        "final_url",
        "published_at",
        "version_label",
        "effective_from",
        "future_effective_from",
        "norm_status",
        "corpus_role",
        "retrieved_at",
        "sha256",
    )
    return {field: entry.get(field) for field in fields if entry.get(field) is not None}


def _extract_pdf(path: Path, metadata: dict[str, Any]) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - mensagem de ambiente
        raise RuntimeError("Instale pypdf para processar os documentos PDF.") from error

    reader = PdfReader(path)
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _normalize_text(page.extract_text() or "")
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={**metadata, "page": page_number},
                )
            )
    return documents


def _extract_html(path: Path, metadata: dict[str, Any]) -> list[Document]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as error:  # pragma: no cover - mensagem de ambiente
        raise RuntimeError("Instale beautifulsoup4 para processar HTML.") from error

    html = path.read_text(encoding="utf-8", errors="replace")
    text = _html_to_text(html, BeautifulSoup)
    if not text:
        return []
    return [Document(page_content=text, metadata={**metadata, "page": None})]


def _html_to_text(html: str, beautiful_soup_class: Any) -> str:
    soup = beautiful_soup_class(html, "html.parser")
    for tag in soup.select("script, style, noscript, svg, nav, header, footer, del, s, strike"):
        tag.decompose()

    for tag in soup.find_all(style=True):
        style = html_module.unescape(str(tag.get("style", ""))).casefold()
        if "line-through" in style:
            tag.decompose()

    content = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    return _normalize_text(content.get_text("\n", strip=True))


def _extract_bcb_normativo_json(path: Path, metadata: dict[str, Any]) -> list[Document]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as error:  # pragma: no cover - mensagem de ambiente
        raise RuntimeError("Instale beautifulsoup4 para processar a norma do BCB.") from error

    payload = _read_json(path)
    entries = payload.get("conteudo")
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
        raise ValueError(f"Resposta normativa do BCB inválida: {path}")
    record = entries[0]
    norm_text = record.get("Texto")
    if not isinstance(norm_text, str) or not norm_text.strip():
        raise ValueError(f"Resposta do BCB sem texto normativo: {path}")

    cleaned_body = _html_to_text(norm_text, BeautifulSoup)
    header_parts = [
        str(record.get("Titulo", "")).strip(),
        str(record.get("Assunto", "")).strip(),
    ]
    footer = str(record.get("DOU", "")).strip()
    text = _normalize_text("\n\n".join(part for part in [*header_parts, cleaned_body, footer] if part))
    enriched_metadata = {
        **metadata,
        "page": None,
        "bcb_record_id": record.get("Id"),
        "bcb_version_number": record.get("VersaoNormativo"),
        "official_title": record.get("Titulo"),
        "official_date_text": record.get("DataTexto"),
        "revoked": bool(record.get("Revogado", False)),
    }
    return [Document(page_content=text, metadata=enriched_metadata)]


def load_corpus_pages(
    manifest_path: Path = DEFAULT_MANIFEST,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> list[Document]:
    """Valida hashes e extrai o corpus bruto em unidades por página/documento."""

    manifest = _read_json(manifest_path)
    entries = manifest.get("documents")
    if not isinstance(entries, list) or not entries:
        raise ValueError("O manifesto deve conter documentos coletados.")

    result: list[Document] = []
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Entrada inválida no manifesto do corpus.")
        document_id = entry.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Documento sem document_id válido.")
        if document_id in seen_ids:
            raise ValueError(f"document_id duplicado no manifesto: {document_id}")
        seen_ids.add(document_id)

        path = _safe_raw_path(raw_dir, entry.get("local_filename"))
        if not path.is_file():
            raise FileNotFoundError(f"Documento do corpus não encontrado: {path}")
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
            raise ValueError(f"Falha de integridade SHA-256 em {document_id}.")

        metadata = _base_metadata(entry)
        document_format = entry.get("format")
        if document_format == "pdf":
            pages = _extract_pdf(path, metadata)
        elif document_format == "html":
            pages = _extract_html(path, metadata)
        elif document_format == "bcb_normativo_json":
            pages = _extract_bcb_normativo_json(path, metadata)
        else:
            raise ValueError(f"Formato não suportado em {document_id}: {document_format!r}")
        if not pages:
            raise ValueError(f"Nenhum texto foi extraído de {document_id}.")
        result.extend(pages)
    return result


def chunk_documents(
    documents: Iterable[Document],
    tokenizer: TokenizerProtocol,
    *,
    chunk_tokens: int = DEFAULT_PARENT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_PARENT_OVERLAP_TOKENS,
) -> list[Document]:
    """Cria chunks pais de 500 tokens/50 de sobreposição por padrão."""

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens deve ser positivo.")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens deve estar entre zero e chunk_tokens - 1.")

    step = chunk_tokens - overlap_tokens
    chunks: list[Document] = []
    for document in documents:
        token_ids = encode_without_truncation(tokenizer, document.page_content)
        if not token_ids:
            continue
        chunk_index = 0
        for token_start in range(0, len(token_ids), step):
            window = token_ids[token_start : token_start + chunk_tokens]
            if not window:
                break
            token_end = token_start + len(window)
            text = _normalize_text(tokenizer.decode(window, skip_special_tokens=True))
            if not text:
                continue

            identity = "|".join(
                (
                    str(document.metadata.get("document_id", "")),
                    str(document.metadata.get("page", "")),
                    str(token_start),
                    str(token_end),
                )
            )
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        **document.metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "token_start": token_start,
                        "token_end": token_end,
                        "token_count": len(window),
                        "chunk_tokens": chunk_tokens,
                        "overlap_tokens": overlap_tokens,
                    },
                )
            )
            chunk_index += 1
            if token_end >= len(token_ids):
                break
    return chunks


def write_documents_jsonl(documents: Iterable[Document], path: Path) -> int:
    """Persiste documentos como JSONL seguro e portável, sem pickle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for document in documents:
            json.dump(
                {"page_content": document.page_content, "metadata": document.metadata},
                file,
                ensure_ascii=False,
            )
            file.write("\n")
            count += 1
    temporary.replace(path)
    return count


def read_documents_jsonl(path: Path) -> list[Document]:
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Linha JSONL inválida em {path}:{line_number}")
            text = value.get("page_content")
            metadata = value.get("metadata")
            if not isinstance(text, str) or not isinstance(metadata, dict):
                raise ValueError(f"Documento inválido em {path}:{line_number}")
            documents.append(Document(page_content=text, metadata=metadata))
    return documents

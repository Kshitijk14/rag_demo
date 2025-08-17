import os 
import json
import gc
import traceback
from tqdm import tqdm
from pathlib import Path
from typing import List
from ..llm.get_llm_func import num_tokens
from utils.helpers.langchain_imports import (
    Document,
    PyMuPDFLoader,
    RecursiveCharacterTextSplitter,
)


def load_docs(base_data_path: Path, logger) -> List[Document]:
    """
    Load all PDFs directly from the given base_data_path (no grouped subfolders).
    """
    try:
        logger.info("[Stage 01, Part 01.1] Loading docs from base directory.....")
        all_docs = []

        if not base_data_path.exists():
            logger.error(f"[Stage 01, Part 01.2] Base directory does not exist: {base_data_path}")
            return []

        # Collect PDFs directly under base_data_path
        pdf_files = list(base_data_path.glob("*.pdf"))
        logger.info(f"[Stage 01, Part 02] Found {len(pdf_files)} PDFs in base directory")

        # Load each PDF
        for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
            try:
                loader = PyMuPDFLoader(str(pdf_file))
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as inner_e:
                logger.warning(f"[Part 02] Failed to load {pdf_file}: {inner_e}")
                logger.debug(traceback.format_exc())
                continue
            finally:
                gc.collect()  # free memory after each file

        logger.info(f"[Stage 01, Part 03] Total documents loaded: {len(all_docs)}")
        return all_docs

    except Exception as e:
        logger.error(f"[Stage 01, Part 03] Error during document loading: {e}")
        logger.debug(traceback.format_exc())
        return []

def flatten_docs(docs: List, logger) -> List[Document]:
    try:
        logger.info("[Stage 01, Part 04] Flattening docs list.....")
        if docs and isinstance(docs[0], list):
            docs_list = [item for sublist in docs for item in sublist]
            logger.info(f"[Stage 01, Part 04(a)] Flattened docs count: {len(docs_list)}")
        else:
            docs_list = docs
            logger.info(f"[Stage 01, Part 04(b)] Docs already flat, count: {len(docs_list)}")
    except Exception as e:
        logger.error(f"[Stage 01, Part 04] Error flattening docs list: {e}")
        logger.debug(traceback.format_exc())
        docs_list = docs  # fallback
    return docs_list

def split_docs(docs: List[Document], chunk_size: int, chunk_overlap: int, logger) -> List[Document]:
    try:
        logger.info("[Stage 01, Part 05] Splitting docs into chunks....")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # length_function=len,
            # is_separator_regex=False,
        )
        logger.info(f"[Stage 01, Part 05.1] Splitting {len(docs)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
        chunks = text_splitter.split_documents(docs)
        
        logger.info(f"[Stage 01, Part 05.2] Total chunks created: {len(chunks)}")
        return chunks
    except Exception as e:
        logger.error(f"[Stage 01, Part 05] Error splitting documents: {e}")
        logger.debug(traceback.format_exc())
        return []

def calc_chunk_ids(chunks, base_data_path, logger):
    try:
        # Page Source : Page Number : Chunk Index
        last_page_id = None
        curr_chunk_idx = 0
        
        logger.info("[Stage 01, Part 06.1.1] Calculating chunk IDs...")
        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            
            # logger.info("[Stage 01, Part 11.1] Normalizing & Standardizing paths, for cross-platform consistency...")
            norm_source = os.path.normpath(source)
            rel_source = os.path.relpath(norm_source, base_data_path).replace("\\", "/")
            
            curr_page_id = (f"{rel_source}:{page}")
            
            # if the page ID is the same as the last one, increment the index
            if curr_page_id == last_page_id:
                curr_chunk_idx += 1
            else:
                curr_chunk_idx = 0
            
            # logger.info("[Stage 01, Part 11.2] Calculating new chunk IDs...")
            chunk_id = (f"{curr_page_id}:{curr_chunk_idx}")
            last_page_id = curr_page_id
            
            # logger.info("[Stage 01, Part 11.3] Adding chunk IDs to metadata...")
            chunk.metadata["chunk_id"] = chunk_id
        
        logger.info("[Stage 01, Part 06.1.2] Chunk IDs calculated successfully.")
        return chunks
    except Exception as e:
        logger.error(f"[Stage 01, Part 06.1] Error calculating chunk IDs: {e}")
        logger.debug(traceback.format_exc())
        return chunks

def load_existing_chunks_metadata(chunks_dir, logger):
    try:
        if not os.path.exists(chunks_dir):
            logger.warning(f"[Stage 01, Part 06.2] No existing chunks file found at {chunks_dir}")
            return set()
        
        existing_ids = set()
        with open(chunks_dir, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunk_data = json.loads(line.strip())
                    chunk_id = chunk_data.get("metadata", {}).get("chunk_id")
                    if chunk_id:
                        existing_ids.add(chunk_id)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"[Stage 01, Part 06.2] Loaded {len(existing_ids)} existing chunk IDs")
        return existing_ids
    except Exception as e:
        logger.error(f"[Stage 01, Part 06.2] Error loading existing chunks metadata: {e}")
        logger.debug(traceback.format_exc())
        return set()

def filter_chunks(chunks: List[Document], lower_limit: int, upper_limit: int, logger) -> List[Document]:
    logger.info("[Stage 01, Part 06.5.1] Applying custom (token-based) filtering on chunks before ingestion...")
    try:
        filtered = []
        for chunk in chunks:
            content = chunk.page_content.strip()
            token_len = num_tokens(content)
            
            # only chunks between lower_limit and upper_limit characters
            if lower_limit < token_len < upper_limit:
                filtered.append(chunk)
            # else:
            #     logger.warning(f"[Stage 01, Part 14.4.2(a)] Filtered out chunk ((tokens={token_len}, len={len(content)}): {chunk.metadata.get('chunk_id')}")
        
        logger.info(f"[Stage 01, Part 06.5.2(a)] Chunks remaining before filter: {len(chunks)}")
        logger.info(f"[Stage 01, Part 06.5.2(b)] Chunks remaining after filter: {len(filtered)}")
        return filtered
    except Exception as e:
        logger.error(f"[Stage 01, Part 06.5] Error applying custom filtering on chunks: {e}")
        logger.debug(traceback.format_exc())
        return chunks

def process_in_batches(documents, ingest_batch_size: int, ingest_fn, logger):
    total = len(documents)
    logger.info(f"[Stage 01, Part 08.3] Processing {total} documents in batches of {ingest_batch_size}...")

    for i in tqdm(range(0, total, ingest_batch_size), desc="Ingesting batches"):
        batch = documents[i:i + ingest_batch_size]
        try:
            ingest_fn(batch)
            logger.info(f"[Stage 01, Part 08.4] Ingested batch {i // ingest_batch_size}")
        except Exception as e:
            logger.error(f"[Stage 01, Part 08.4] Failed to ingest batch {i // ingest_batch_size}: {e}")
            logger.debug(traceback.format_exc())

def save_processed_chunks_metadata(chunks, chunks_dir, logger):
    # Save successfully processed chunks metadata to JSON file for tracking
    try:
        logger.info(f"[Stage 01, Part 10.1.1] Saving processed chunks metadata to: {chunks_dir}")
        chunks_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Read existing data first
        existing_data = []
        if os.path.exists(chunks_dir):
            with open(chunks_dir, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        existing_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        
        # Append new chunks
        with open(chunks_dir, "a", encoding="utf-8") as f:
            for chunk in chunks:
                chunk_data = {
                    "content": chunk.page_content, 
                    "metadata": chunk.metadata
                }
                f.write(json.dumps(chunk_data) + "\n")
        
        logger.info(f"[Stage 01, Part 10.1.2] Saved {len(chunks)} processed chunks to metadata file")
        
    except Exception as e:
        logger.error(f"[Stage 01, Part 10.1] Error saving processed chunks metadata: {e}")
        logger.debug(traceback.format_exc())
    
    # Load existing chunks metadata from JSON file to track what's already processed
    try:
        if not os.path.exists(chunks_dir):
            logger.info(f"[Stage 01, Part 10.2.1] No existing chunks file found at {chunks_dir}")
            return set()
        
        existing_ids = set()
        with open(chunks_dir, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunk_data = json.loads(line.strip())
                    chunk_id = chunk_data.get("metadata", {}).get("chunk_id")
                    if chunk_id:
                        existing_ids.add(chunk_id)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"[Stage 01, Part 10.2.2] Loaded {len(existing_ids)} existing chunk IDs")
        return existing_ids
    except Exception as e:
        logger.error(f"[Stage 01, Part 10.2] Error loading existing chunks metadata: {e}")
        logger.debug(traceback.format_exc())
        return set()
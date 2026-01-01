#!/usr/bin/env python3
"""
vectorize_database.py
Enhanced version with incremental updates - only updates changed products
"""

import os
import sys
import time
import json
import re
import uuid
import math
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import pymysql
from dotenv import load_dotenv

# LangChain HuggingFace wrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Pinecone (serverless)
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# -------------------------
# Configuration & Logging
# -------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT") or 3306),
    "charset": "utf8mb4"
}

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "products-index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIMENSION = 384
BATCH_SIZE = int(os.getenv("PINECONE_BATCH_SIZE") or 50)
UPSERT_SLEEP = float(os.getenv("UPSERT_SLEEP") or 0.15)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("vectorize_database")

# -------------------------
# Sentiment Analysis Helper
# -------------------------
def analyze_sentiment(text: str) -> str:
    """
    Simple rule-based sentiment analysis for Bangla and English reviews
    Returns: 'positive', 'negative', or 'neutral'
    """
    if not text:
        return "neutral"
    
    text_lower = text.lower()
    
    # Positive indicators (Bangla + English)
    positive_indicators = [
        # English
        'good', 'great', 'excellent', 'awesome', 'amazing', 'perfect', 'love', 'nice', 
        'best', 'wonderful', 'fantastic', 'outstanding', 'superb', 'brilliant',
        # Bangla
        'ভাল', 'চমৎকার', 'অসাধারণ', 'দারুণ', 'বেশ', 'মজা', 'সুন্দর', 'উত্তম', 'খুব ভাল',
        'ভালো', 'বেশ ভালো', 'অনেক ভালো', 'খুশি', 'সন্তুষ্ট'
    ]
    
    # Negative indicators (Bangla + English)
    negative_indicators = [
        # English
        'bad', 'poor', 'terrible', 'awful', 'horrible', 'worst', 'disappointing',
        'waste', 'cheap', 'broken', 'damaged', 'problem', 'issue', 'complaint',
        # Bangla
        'খারাপ', 'ভাল না', 'খারাপ লাগছে', 'অসন্তুষ্ট', 'সমস্যা', 'ত্রুটি', 'ভাঙ্গা',
        'নষ্ট', 'কম', 'দুঃখিত', 'অসুবিধা', 'অভিযোগ'
    ]
    
    positive_count = sum(1 for word in positive_indicators if word in text_lower)
    negative_count = sum(1 for word in negative_indicators if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment"""
    return {
        "positive": "😊",
        "negative": "😞", 
        "neutral": "😐"
    }.get(sentiment, "😐")

# -------------------------
# Utilities: HTML -> Text
# -------------------------
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_WHITESPACE = re.compile(r"\s+")
RE_ENTITY = re.compile(r"&[A-Za-z0-9#]+;")
RE_ALLOWED = re.compile(r"[^\w\s\u0980-\u09FF\.\,\!\?\-\:\;\(\)\%\$\@\#\&\—\'\"]+")

def clean_html_completely(raw_html: Optional[str]) -> str:
    if not raw_html:
        return ""
    text = raw_html
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = RE_HTML_TAG.sub(" ", text)
    text = RE_ENTITY.sub(" ", text)
    text = RE_ALLOWED.sub(" ", text)
    text = RE_WHITESPACE.sub(" ", text).strip()
    return text

def intelligently_pick_summary(text: str, tags: List[str], name: str, max_chars: int = 320) -> str:
    if not text:
        base = f"{name}".strip()
    else:
        sentences = re.split(r'(?<=[।\.!\?])\s+', text)
        meaningful = [s.strip() for s in sentences if len(s.strip()) > 20]
        if meaningful:
            base = " ".join(meaningful[:2])
        else:
            base = text[:max_chars]

    tag_hint = ""
    top_tags = [t for t in tags if isinstance(t, str) and t.strip()][:6]
    if top_tags:
        tag_hint = " Tags: " + ", ".join(top_tags[:6])

    summary = f"{base}{tag_hint}"
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    return summary

# -------------------------
# Safe conversions
# -------------------------
def safe_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (bytes, bytearray)):
        try:
            if len(value) == 16:
                return str(uuid.UUID(bytes=value))
            else:
                return value.decode('utf-8', errors='ignore').strip()
        except Exception:
            return default
    try:
        return str(value)
    except Exception:
        return default

def safe_uuid(value: Any) -> str:
    if not value:
        return str(uuid.uuid4())
    if isinstance(value, uuid.UUID):
        return str(value)
    str_value = safe_string(value)
    if not str_value:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(str_value))
    except ValueError:
        pass
    if isinstance(value, (bytes, bytearray)):
        try:
            if len(value) == 16:
                return str(uuid.UUID(bytes=value))
            else:
                hex_str = value.hex()
                if len(hex_str) == 32:
                    return str(uuid.UUID(hex_str))
        except Exception:
            pass
    return str(uuid.uuid4())

def safe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [safe_string(x) for x in value if safe_string(x)]
    if isinstance(value, (str, bytes)):
        s = value.decode() if isinstance(value, bytes) else value
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [safe_string(x) for x in parsed if safe_string(x)]
            except Exception:
                pass
        if "," in s:
            parts = [p.strip().strip('"\'') for p in s.split(",") if p.strip()]
            return [p for p in parts if p]
        return [s] if s else []
    return [safe_string(value)]

def cast_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default

def cast_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse datetime string to datetime object"""
    if not dt_str:
        return None
    try:
        # Handle different datetime formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S.%f"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        return None
    except Exception:
        return None

# -------------------------
# Database functions
# -------------------------
def get_db_connection():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error("DB connection failed: %s", e)
        raise

def fetch_all_products(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = []
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            q = """
            SELECT uid, name, description, price, brand, category, quantity,
                   discount, size, weight, color, tags, free_delivery, best_selling,
                   created_at, updated_at
            FROM Products
            ORDER BY created_at DESC
            """
            if limit:
                cur.execute(q + " LIMIT %s", (limit,))
            else:
                cur.execute(q)
            rows = cur.fetchall()
            logger.info("Fetched %d product rows", len(rows))
    except Exception as e:
        logger.exception("Error fetching products: %s", e)
    finally:
        conn.close()
    return rows

def get_product_reviews(conn, product_uid: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get reviews with proper BINARY(16) UUID handling
    """
    try:
        # Convert string UUID to bytes for BINARY(16) column
        product_uid_bytes = uuid.UUID(product_uid).bytes
        
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(
                """
                SELECT rating, review_text, user_name, created_at
                FROM ProductReview
                WHERE product_uid = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (product_uid_bytes, limit),
            )
            results = cur.fetchall()
            return results
    except Exception as e:
        logger.error("Error fetching reviews for %s: %s", product_uid, e)
        return []

def rating_stats_from_reviews(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not reviews:
        return {"average": 0.0, "count": 0}
    
    valid_ratings = []
    for r in reviews:
        rating = r.get("rating")
        if rating is not None:
            try:
                rating_val = float(rating)
                if 0 <= rating_val <= 5:
                    valid_ratings.append(rating_val)
            except (ValueError, TypeError):
                continue
    
    if not valid_ratings:
        return {"average": 0.0, "count": 0}
    
    avg = round(sum(valid_ratings) / len(valid_ratings), 1)
    return {"average": avg, "count": len(valid_ratings)}

def format_reviews_for_preview(reviews: List[Dict[str, Any]]) -> str:
    """
    Format reviews with names, ratings, sentiment, and emojis for preview
    """
    if not reviews:
        return "No reviews yet"
    
    formatted_reviews = []
    for i, review in enumerate(reviews[:5]):  # Show max 5 reviews in preview
        user_name = review.get('user_name', 'Anonymous')
        rating = review.get('rating', 0)
        review_text = review.get('review_text', '')
        
        # Analyze sentiment
        sentiment = analyze_sentiment(review_text)
        emoji = get_sentiment_emoji(sentiment)
        
        # Truncate long review text
        truncated_text = review_text[:150] + "..." if len(review_text) > 150 else review_text
        
        formatted_review = f"{emoji} {user_name}: ⭐{rating}/5 ({sentiment}) - {truncated_text}"
        formatted_reviews.append(formatted_review)
    
    return "\n".join(formatted_reviews)

# -------------------------
# Pinecone Helper Functions
# -------------------------
def get_existing_vector_timestamps(index) -> Dict[str, datetime]:
    """
    Fetch current updated_at timestamps from Pinecone for all vectors
    """
    timestamps = {}
    try:
        # Fetch all vectors (this might be expensive for large indexes)
        # Consider using pagination if you have many vectors
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        
        if total_vectors > 0:
            # Fetch vectors in batches
            logger.info("Fetching existing vector timestamps from Pinecone...")
            # Note: This is a simplified approach. For large datasets, you might need pagination
            # or store the timestamps separately
            
    except Exception as e:
        logger.warning("Could not fetch existing vector timestamps: %s", e)
    
    return timestamps

def get_vector_metadata(index, vector_id: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific vector
    """
    try:
        result = index.fetch(ids=[vector_id])
        if vector_id in result.vectors:
            return result.vectors[vector_id].metadata
    except Exception as e:
        logger.debug("Could not fetch metadata for vector %s: %s", vector_id, e)
    return None

def has_product_changed(db_product: Dict[str, Any], existing_metadata: Optional[Dict[str, Any]]) -> bool:
    """
    Check if product has changed by comparing updated_at timestamps
    """
    if not existing_metadata:
        return True  # No existing vector, needs to be created
    
    # Get current updated_at from database
    db_updated_at_str = safe_string(db_product.get("updated_at"))
    db_updated_at = parse_datetime(db_updated_at_str)
    
    # Get existing updated_at from Pinecone
    existing_updated_at_str = existing_metadata.get("updated_at")
    existing_updated_at = parse_datetime(existing_updated_at_str)
    
    if not db_updated_at or not existing_updated_at:
        return True  # If we can't parse dates, assume changed
    
    # Check if database timestamp is newer than Pinecone timestamp
    return db_updated_at > existing_updated_at

# -------------------------
# Create embedding document
# -------------------------
def create_product_document(product: Dict[str, Any], reviews: List[Dict[str, Any]]) -> Document:
    uid = safe_uuid(product.get("uid"))
    name = safe_string(product.get("name", "Unknown Product"))
    raw_description = safe_string(product.get("description", ""))
    description = clean_html_completely(raw_description)
    category = safe_string(product.get("category", ""))
    brand = safe_string(product.get("brand", ""))
    price = cast_float(product.get("price", 0.0))
    discount = cast_float(product.get("discount", 0.0))
    quantity = cast_int(product.get("quantity", 0))
    sizes = safe_list(product.get("size"))
    weights = safe_list(product.get("weight"))
    colors = safe_list(product.get("color"))
    tags = safe_list(product.get("tags"))
    free_delivery = bool(product.get("free_delivery"))
    best_selling = bool(product.get("best_selling"))
    created_at = safe_string(product.get("created_at", ""))
    updated_at = safe_string(product.get("updated_at", ""))

    # Reviews & rating
    stats = rating_stats_from_reviews(reviews)
    review_texts = [safe_string(r.get("review_text", "")) for r in reviews if safe_string(r.get("review_text", ""))]
    review_snippet = " ".join(review_texts[:3]) if review_texts else ""
    
    # Format reviews for preview with names, ratings, and sentiment
    formatted_reviews = format_reviews_for_preview(reviews)

    summary = intelligently_pick_summary(description or review_snippet or name, tags, name)

    parts = [
        f"Product: {name}",
        f"Category: {category}" if category else "",
        f"Brand: {brand}" if brand else "",
        f"Price: {price:.2f} BDT" if price else "",
        f"Discount: {discount}%" if discount else "",
        f"Overall Rating: {stats['average']}/5 ({stats['count']} reviews)",
        f"Summary: {summary}",
        f"Description: {description}" if description else "",
    ]

    attr_parts = []
    if sizes:
        attr_parts.append("Sizes: " + ", ".join(sizes[:10]))
    if weights:
        attr_parts.append("Weights: " + ", ".join(weights[:10]))
    if colors:
        attr_parts.append("Colors: " + ", ".join(colors[:10]))
    if tags:
        attr_parts.append("Tags: " + ", ".join(tags[:15]))
    if attr_parts:
        parts.append("Attributes: " + " | ".join(attr_parts))

    # Enhanced reviews section with names, ratings, and sentiment
    if reviews:
        parts.append("Customer Reviews:")
        parts.append(formatted_reviews)

    if free_delivery:
        parts.append("Free Delivery Available")
    if best_selling:
        parts.append("Best Selling Product")

    document_text = "\n".join([p for p in parts if p]).strip()

    metadata = {
        "product_id": uid,
        "name": name,
        "category": category,
        "brand": brand,
        "price": price,
        "discount": discount,
        "quantity": quantity,
        "average_rating": stats["average"],
        "review_count": stats["count"],
        "free_delivery": free_delivery,
        "best_selling": best_selling,
        "sizes": sizes,
        "weights": weights,
        "colors": colors,
        "tags": tags,
        "created_at": created_at,
        "updated_at": updated_at,
        "source": "database",
        "content_type": "product"
    }

    return Document(page_content=document_text, metadata=metadata)

# -------------------------
# Initialize embeddings and Pinecone
# -------------------------
def init_embeddings():
    logger.info("Initializing embeddings model: %s", EMBEDDING_MODEL)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        test_emb = embeddings.embed_query("test")
        logger.info("Embeddings initialized (dim=%d)", len(test_emb))
        return embeddings
    except Exception as e:
        logger.exception("Failed to initialize embeddings: %s", e)
        raise

def init_pinecone(delete_existing: bool = False):
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY missing in env")
        raise RuntimeError("PINECONE_API_KEY missing")

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = pc.list_indexes()
        index_names = [idx.name for idx in existing.indexes] if getattr(existing, "indexes", None) else []
        
        if delete_existing and PINECONE_INDEX_NAME in index_names:
            logger.info("Deleting existing Pinecone index: %s", PINECONE_INDEX_NAME)
            pc.delete_index(PINECONE_INDEX_NAME)
            logger.info("Waiting for index deletion...")
            time.sleep(10)
            index_names.remove(PINECONE_INDEX_NAME)
        
        if PINECONE_INDEX_NAME not in index_names:
            logger.info("Creating Pinecone index: %s (dim=%d)", PINECONE_INDEX_NAME, EMBED_DIMENSION)
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBED_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info("Waiting for index creation...")
            time.sleep(10)
        else:
            logger.info("Using existing Pinecone index: %s", PINECONE_INDEX_NAME)

        index = pc.Index(PINECONE_INDEX_NAME)
        return index
    except Exception as e:
        logger.exception("Error initializing Pinecone: %s", e)
        raise

def upsert_batch(index, batch_vectors: List[Dict[str, Any]]):
    if not batch_vectors:
        return
    try:
        index.upsert(vectors=batch_vectors)
        logger.debug("Upserted batch size=%d", len(batch_vectors))
    except Exception as e:
        logger.exception("Batch upsert failed: %s", e)
        for v in batch_vectors:
            try:
                index.upsert(vectors=[v])
                time.sleep(UPSERT_SLEEP / 4)
            except Exception:
                logger.exception("Failed to upsert vector id=%s", v.get("id"))

def vectorize_all_products(limit: Optional[int] = None, delete_existing_index: bool = False, incremental: bool = True):
    logger.info("Starting vectorization pipeline (incremental=%s)", incremental)
    embeddings = init_embeddings()
    index = init_pinecone(delete_existing=delete_existing_index)
    products = fetch_all_products(limit=limit)
    if not products:
        logger.warning("No products to process, exiting.")
        return

    conn = get_db_connection()
    batch = []
    processed = 0
    updated = 0
    skipped = 0
    start_time = time.time()

    try:
        for row in products:
            try:
                uid = safe_uuid(row.get("uid"))
                if not uid:
                    logger.warning("Skipping product with missing uid: %s", row.get("name"))
                    continue

                vector_id = f"prod_{uid}"
                
                # Check if we need to update this product (incremental mode)
                if incremental:
                    existing_metadata = get_vector_metadata(index, vector_id)
                    if not has_product_changed(row, existing_metadata):
                        logger.debug("Skipping unchanged product: %s (UID: %s)", row.get("name"), uid)
                        skipped += 1
                        continue

                logger.info("Processing product: %s (UID: %s)", row.get("name"), uid)
                
                reviews = get_product_reviews(conn, uid, limit=50)
                logger.info("Found %d reviews for product %s", len(reviews), uid)
                
                doc = create_product_document(row, reviews)
                emb = embeddings.embed_query(doc.page_content)
                emb = [float(x) for x in emb]

                vector_meta = doc.metadata.copy()
                vector_meta["_preview"] = doc.page_content

                vect = {
                    "id": vector_id,
                    "values": emb,
                    "metadata": vector_meta
                }

                batch.append(vect)
                processed += 1
                updated += 1

                if len(batch) >= BATCH_SIZE:
                    upsert_batch(index, batch)
                    batch = []
                    time.sleep(UPSERT_SLEEP)

                if processed % 5 == 0:
                    logger.info("Processed %d products (updated: %d, skipped: %d)", 
                               processed, updated, skipped)
                    if reviews:
                        logger.info("Sample - Product: %s, Reviews: %d, Avg Rating: %.1f", 
                                   row.get("name"), len(reviews), 
                                   doc.metadata.get("average_rating", 0))

            except Exception as e:
                logger.exception("Error processing product uid=%s: %s", uid, e)
                continue

        if batch:
            upsert_batch(index, batch)

        try:
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count if hasattr(stats, "total_vector_count") else "unknown"
            logger.info("Vectorization complete: processed=%d, updated=%d, skipped=%d, index_total_vectors=%s, elapsed=%.1fs",
                        processed, updated, skipped, total_vectors, time.time() - start_time)
        except Exception as e:
            logger.warning("Could not fetch index stats: %s", e)

    finally:
        try:
            conn.close()
        except Exception:
            pass

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vectorize products and upload to Pinecone")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of products to process")
    parser.add_argument("--batch-size", type=int, default=None, help="Pinecone upsert batch size")
    parser.add_argument("--delete-index", action="store_true", help="Delete existing index before starting")
    parser.add_argument("--full-sync", action="store_true", help="Force full sync (ignore incremental updates)")
    args = parser.parse_args()

    current_batch_size = BATCH_SIZE
    if args.batch_size:
        current_batch_size = args.batch_size
        logger.info("Using batch size: %d", current_batch_size)

    vectorize_all_products(
        limit=args.limit, 
        delete_existing_index=args.delete_index,
        incremental=not args.full_sync
    )

if __name__ == "__main__":
    main()
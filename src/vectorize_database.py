#!/usr/bin/env python3
"""
Vectorize database products to Pinecone with 768-dimension embeddings
"""

import os
import sys
import time
import json
import re
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import pymysql
from dotenv import load_dotenv

# LangChain imports with fallbacks
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
    

# Pinecone
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# -------------------------
# Configuration
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
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") or "ecommerce-customer-support"  # Updated to match your config
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-mpnet-base-v2"
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION") or 768)
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
# Helper Functions
# -------------------------
def clean_html(text: Optional[str]) -> str:
    """Clean HTML tags from text"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def safe_uuid(value: Any) -> str:
    """Safely convert to UUID string"""
    if not value:
        return str(uuid.uuid4())
    
    if isinstance(value, uuid.UUID):
        return str(value)
    
    if isinstance(value, (bytes, bytearray)):
        try:
            if len(value) == 16:
                return str(uuid.UUID(bytes=value))
        except Exception:
            pass
    
    str_value = str(value).strip()
    try:
        return str(uuid.UUID(str_value))
    except ValueError:
        return str(uuid.uuid4())


def safe_string(value: Any, default: str = "") -> str:
    """Safely convert to string"""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    try:
        return str(value).strip()
    except Exception:
        return default


def safe_list(value: Any) -> List[str]:
    """Safely convert to list of strings"""
    if value is None:
        return []
    if isinstance(value, list):
        return [safe_string(x) for x in value if safe_string(x)]
    if isinstance(value, str):
        if value.startswith('[') and value.endswith(']'):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [safe_string(x) for x in parsed if safe_string(x)]
            except Exception:
                pass
        # Split by comma if it looks like a comma-separated list
        if ',' in value:
            parts = [p.strip() for p in value.split(',') if p.strip()]
            return parts
        return [value.strip()] if value.strip() else []
    return [safe_string(value)]


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert to int"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


# -------------------------
# Database Functions
# -------------------------
def get_db_connection():
    """Create database connection"""
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


def fetch_all_products(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch all products from database"""
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            query = """
            SELECT uid, name, description, price, brand, category, quantity,
                   discount, size, weight, color, tags, free_delivery, best_selling,
                   created_at, updated_at
            FROM Products
            ORDER BY created_at DESC
            """
            if limit:
                cursor.execute(query + " LIMIT %s", (limit,))
            else:
                cursor.execute(query)
            
            rows = cursor.fetchall()
            logger.info(f"Fetched {len(rows)} products from database")
            return rows
    except Exception as e:
        logger.error(f"Error fetching products: {e}")
        return []
    finally:
        conn.close()


def get_product_reviews(conn, product_uid: str) -> List[Dict[str, Any]]:
    """Get reviews for a product"""
    try:
        # Convert UUID string to bytes for database
        product_uuid = uuid.UUID(product_uid)
        product_uid_bytes = product_uuid.bytes
        
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT rating, review_text, user_name, created_at
                FROM ProductReview
                WHERE product_uid = %s
                ORDER BY created_at DESC
                LIMIT 20
            """, (product_uid_bytes,))
            
            reviews = cursor.fetchall()
            return reviews
    except Exception as e:
        logger.error(f"Error fetching reviews for {product_uid}: {e}")
        return []


def calculate_rating_stats(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate average rating from reviews"""
    if not reviews:
        return {"average": 0.0, "count": 0}
    
    valid_ratings = []
    for review in reviews:
        rating = review.get("rating")
        if rating is not None:
            try:
                rating_val = float(rating)
                if 0 <= rating_val <= 5:
                    valid_ratings.append(rating_val)
            except (ValueError, TypeError):
                continue
    
    if not valid_ratings:
        return {"average": 0.0, "count": 0}
    
    average = sum(valid_ratings) / len(valid_ratings)
    return {"average": round(average, 1), "count": len(valid_ratings)}


# -------------------------
# Document Creation
# -------------------------
def create_product_document(product: Dict[str, Any], reviews: List[Dict[str, Any]]) -> Document:
    """Create LangChain Document from product data"""
    # Extract product data
    product_id = safe_uuid(product.get("uid"))
    name = safe_string(product.get("name", "Unknown Product"))
    description = clean_html(safe_string(product.get("description", "")))
    brand = safe_string(product.get("brand", ""))
    category = safe_string(product.get("category", ""))
    price = safe_float(product.get("price", 0.0))
    discount = safe_float(product.get("discount", 0.0))
    quantity = safe_int(product.get("quantity", 0))
    tags = safe_list(product.get("tags", ""))
    sizes = safe_list(product.get("size", ""))
    colors = safe_list(product.get("color", ""))
    weights = safe_list(product.get("weight", ""))
    free_delivery = bool(product.get("free_delivery"))
    best_selling = bool(product.get("best_selling"))
    created_at = safe_string(product.get("created_at", ""))
    updated_at = safe_string(product.get("updated_at", ""))
    
    # Calculate rating stats
    rating_stats = calculate_rating_stats(reviews)
    
    # Build document text
    parts = [
        f"Product: {name}",
        f"Brand: {brand}" if brand else "",
        f"Category: {category}" if category else "",
        f"Price: {price:.2f} BDT" if price > 0 else "",
        f"Discount: {discount}%" if discount > 0 else "",
        f"Rating: {rating_stats['average']}/5 ({rating_stats['count']} reviews)" if rating_stats['count'] > 0 else "",
    ]
    
    if description:
        parts.append(f"Description: {description[:300]}")
    
    if tags:
        parts.append(f"Tags: {', '.join(tags[:10])}")
    
    if sizes:
        parts.append(f"Sizes: {', '.join(sizes[:5])}")
    
    if colors:
        parts.append(f"Colors: {', '.join(colors[:5])}")
    
    if free_delivery:
        parts.append("Free Delivery Available")
    
    if best_selling:
        parts.append("Best Selling Product")
    
    # Add sample reviews
    if reviews and len(reviews) > 0:
        parts.append("Recent Reviews:")
        for i, review in enumerate(reviews[:3]):
            review_text = safe_string(review.get("review_text", ""))
            rating = safe_float(review.get("rating", 0))
            user = safe_string(review.get("user_name", "Customer"))
            if review_text:
                parts.append(f"- {user}: ⭐{rating}/5 - {review_text[:100]}")
    
    document_text = "\n".join([p for p in parts if p])
    
    # Build metadata
    metadata = {
        "product_id": product_id,
        "name": name,
        "brand": brand,
        "category": category,
        "price": price,
        "discount": discount,
        "quantity": quantity,
        "average_rating": rating_stats["average"],
        "review_count": rating_stats["count"],
        "tags": tags,
        "sizes": sizes,
        "colors": colors,
        "weights": weights,
        "free_delivery": free_delivery,
        "best_selling": best_selling,
        "created_at": created_at,
        "updated_at": updated_at,
        "source": "database",
        "content_type": "product"
    }
    
    return Document(page_content=document_text, metadata=metadata)


# -------------------------
# Pinecone Functions
# -------------------------
def init_pinecone(delete_existing: bool = False):
    """Initialize Pinecone connection"""
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY not found in environment")
        raise ValueError("PINECONE_API_KEY is required")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check existing indexes
    existing_indexes = pc.list_indexes()
    index_names = [idx.name for idx in existing_indexes.indexes]
    
    # Delete existing index if requested
    if delete_existing and PINECONE_INDEX_NAME in index_names:
        logger.info(f"Deleting existing index: {PINECONE_INDEX_NAME}")
        pc.delete_index(PINECONE_INDEX_NAME)
        time.sleep(10)
        index_names.remove(PINECONE_INDEX_NAME)
    
    # Create index if it doesn't exist
    if PINECONE_INDEX_NAME not in index_names:
        logger.info(f"Creating index: {PINECONE_INDEX_NAME} (dim={EMBED_DIMENSION})")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info("Waiting for index creation...")
        time.sleep(15)
    
    # Get index
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Verify index
    try:
        stats = index.describe_index_stats()
        logger.info(f"Pinecone index ready. Total vectors: {stats.total_vector_count}")
        return index
    except Exception as e:
        logger.error(f"Error connecting to Pinecone index: {e}")
        raise


def upsert_batch(index, vectors: List[Dict], batch_num: int):
    """Upsert a batch of vectors to Pinecone"""
    if not vectors:
        return
    
    try:
        index.upsert(vectors=vectors)
        logger.info(f"Batch {batch_num}: Upserted {len(vectors)} vectors")
    except Exception as e:
        logger.error(f"Error upserting batch {batch_num}: {e}")


# -------------------------
# Main Vectorization Function
# -------------------------
def vectorize_products(limit: Optional[int] = None, delete_index: bool = False, batch_size: int = BATCH_SIZE):
    """Main vectorization pipeline"""
    logger.info("Starting vectorization pipeline")
    logger.info(f"Embedding model: {EMBEDDING_MODEL} (dim={EMBED_DIMENSION})")
    logger.info(f"Pinecone index: {PINECONE_INDEX_NAME}")
    
    # Initialize embeddings
    logger.info("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Initialize Pinecone
    logger.info("Initializing Pinecone...")
    index = init_pinecone(delete_existing=delete_index)
    
    # Fetch products
    logger.info("Fetching products from database...")
    products = fetch_all_products(limit=limit)
    
    if not products:
        logger.warning("No products found in database")
        return
    
    logger.info(f"Processing {len(products)} products...")
    
    # Database connection for reviews
    db_conn = get_db_connection()
    
    # Process products
    vectors = []
    processed = 0
    start_time = time.time()
    
    for product in products:
        try:
            product_id = safe_uuid(product.get("uid"))
            if not product_id:
                logger.warning(f"Skipping product with invalid UUID: {product.get('name')}")
                continue
            
            logger.info(f"Processing: {product.get('name', 'Unknown')} ({product_id})")
            
            # Get reviews
            reviews = get_product_reviews(db_conn, product_id)
            if reviews:
                logger.debug(f"Found {len(reviews)} reviews")
            
            # Create document
            document = create_product_document(product, reviews)
            
            # Generate embedding
            embedding = embeddings.embed_query(document.page_content)
            
            # Create vector
            vector = {
                "id": f"prod_{product_id}",
                "values": embedding,
                "metadata": {
                    **document.metadata,
                    "_preview": document.page_content[:500]  # Store preview for search
                }
            }
            
            vectors.append(vector)
            processed += 1
            
            # Upsert in batches
            if len(vectors) >= batch_size:
                batch_num = processed // batch_size
                upsert_batch(index, vectors, batch_num)
                vectors = []
                time.sleep(UPSERT_SLEEP)
            
            # Log progress
            if processed % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Processed {processed}/{len(products)} products ({elapsed:.1f}s)")
                
        except Exception as e:
            logger.error(f"Error processing product {product.get('name')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Upsert remaining vectors
    if vectors:
        upsert_batch(index, vectors, "final")
    
    # Get final stats
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        logger.info(f"Vectorization complete!")
        logger.info(f"Total products processed: {processed}")
        logger.info(f"Total vectors in index: {total_vectors}")
        logger.info(f"Total time: {time.time() - start_time:.1f}s")
        
    except Exception as e:
        logger.error(f"Error getting final stats: {e}")
    
    finally:
        db_conn.close()


# -------------------------
# Main Function
# -------------------------
def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vectorize products to Pinecone")
    parser.add_argument("--limit", type=int, help="Limit number of products to process")
    parser.add_argument("--delete-index", action="store_true", 
                       help="Delete existing Pinecone index before starting")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help=f"Batch size for upserts (default: {BATCH_SIZE})")
    
    args = parser.parse_args()
    
    logger.info(f"Starting vectorization with config:")
    logger.info(f"  Model: {EMBEDDING_MODEL}")
    logger.info(f"  Dimension: {EMBED_DIMENSION}")
    logger.info(f"  Pinecone Index: {PINECONE_INDEX_NAME}")
    logger.info(f"  Batch Size: {args.batch_size}")
    
    try:
        vectorize_products(
            limit=args.limit,
            delete_index=args.delete_index,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Vectorization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
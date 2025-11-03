"""
Manual vectorization script to properly upload MySQL data to Pinecone
Run this script to populate your Pinecone vector store
"""
import os
import sys
import pymysql
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
import uuid
import time
import re

load_dotenv()

# Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '43.204.142.93'),
    'user': os.getenv('DB_USER', 'mahbubdb'),
    'password': os.getenv('DB_PASSWORD', 'mahbub123'),
    'database': os.getenv('DB_NAME', 'ecommerce_website'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'charset': 'utf8mb4'
}

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ecommerce-customer-support")

def clean_html(text):
    """Remove HTML tags and clean up whitespace from text"""
    if not text:
        return ""
    
    # Remove HTML tags
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    
    # Remove extra whitespace, newlines, and special characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = re.sub(r'\r\n', ' ', text)  # Replace Windows newlines
    text = re.sub(r'\n', ' ', text)    # Replace Unix newlines
    text = re.sub(r'\t', ' ', text)    # Replace tabs
    text = re.sub(r'\r', ' ', text)    # Replace carriage returns
    
    # Remove any remaining special characters but keep Bengali and English text
    text = re.sub(r'[^\w\s\u0980-\u09FF\.\,\!\?\-\$]', '', text)
    
    return text.strip()

def extract_clean_text(html_text, max_length=300):
    """Extract clean, readable text from HTML"""
    if not html_text:
        return ""
    
    # Remove HTML tags and clean
    clean_text = clean_html(html_text)
    
    # Extract the most meaningful parts (usually the first few sentences)
    sentences = re.split(r'[.!?]+', clean_text)
    meaningful_text = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:  # Only take substantial sentences
            meaningful_text += sentence + ". "
            if len(meaningful_text) > max_length:
                break
    
    return meaningful_text.strip() or clean_text[:max_length]

def init_embeddings():
    """Initialize HuggingFace embeddings"""
    print("🔄 Initializing embeddings model...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        # Test the embeddings
        test_embedding = embeddings.embed_query("test")
        print(f"✅ Embeddings model loaded successfully (dimension: {len(test_embedding)})")
        return embeddings
    except Exception as e:
        print(f"❌ ERROR loading embeddings: {e}")
        sys.exit(1)

def init_pinecone():
    """Initialize Pinecone connection"""
    if not PINECONE_API_KEY:
        print("❌ ERROR: PINECONE_API_KEY not set")
        sys.exit(1)
    
    print("🔄 Connecting to Pinecone...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes.indexes]
        
        if PINECONE_INDEX_NAME not in index_names:
            print(f"📦 Creating Pinecone index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("⏳ Index created. Waiting for initialization...")
            time.sleep(30)
        else:
            print(f"✅ Pinecone index {PINECONE_INDEX_NAME} already exists")
        
        index = pc.Index(PINECONE_INDEX_NAME)
        print(f"✅ Connected to index: {PINECONE_INDEX_NAME}")
        return index
    
    except Exception as e:
        print(f"❌ ERROR initializing Pinecone: {e}")
        sys.exit(1)

def get_db_connection():
    """Get MySQL connection"""
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ ERROR connecting to database: {e}")
        sys.exit(1)

def fetch_products():
    """Fetch all products from database"""
    connection = get_db_connection()
    products = []
    
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT uid, name, description, price, brand, category, quantity
                FROM Products
            """)
            products = cursor.fetchall()
            print(f"✅ Fetched {len(products)} products from database")
            
            # Show sample products
            for i, product in enumerate(products[:3]):
                print(f"   Sample {i+1}: {product['name'][:50]}...")
                
    except Exception as e:
        print(f"❌ ERROR fetching products: {e}")
    finally:
        connection.close()
    
    return products

def fetch_reviews():
    """Fetch all reviews from database"""
    connection = get_db_connection()
    reviews = []
    
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT rating, review_text, product_uid, user_name
                FROM ProductReview
            """)
            reviews = cursor.fetchall()
            print(f"✅ Fetched {len(reviews)} reviews from database")
    except Exception as e:
        print(f"❌ ERROR fetching reviews: {e}")
    finally:
        connection.close()
    
    return reviews

def create_documents(products, reviews):
    """Create Document objects from products and reviews"""
    documents = []
    
    # Create product documents
    for product in products:
        try:
            product_id = str(uuid.UUID(bytes=product['uid'])) if isinstance(product['uid'], bytes) else str(product['uid'])
            
            # Clean HTML from description and extract meaningful text
            clean_description = extract_clean_text(product.get('description', ''), 200)
            
            product_text = f"""Product: {product['name']}
Brand: {product.get('brand', 'Not specified')}
Category: {product.get('category', 'Not specified')}
Price: ${product.get('price', 0)}
Description: {clean_description}
Available Quantity: {product.get('quantity', 0)} units"""
            
            documents.append(Document(
                page_content=product_text,
                metadata={
                    "type": "product",
                    "product_id": product_id,
                    "name": product['name'],
                    "category": product.get('category', ''),
                    "price": str(product.get('price', 0)),
                    "source": "product"
                }
            ))
        except Exception as e:
            print(f"⚠️ Error creating product document: {e}")
            continue
    
    # Create review documents
    for review in reviews:
        try:
            product_id = str(uuid.UUID(bytes=review['product_uid'])) if isinstance(review['product_uid'], bytes) else str(review['product_uid'])
            
            # Clean review text
            clean_review = clean_html(review.get('review_text', ''))
            
            review_text = f"""Review Rating: {review['rating']}/5
Reviewer: {review['user_name']}
Comment: {clean_review}"""
            
            documents.append(Document(
                page_content=review_text,
                metadata={
                    "type": "review",
                    "product_id": product_id,
                    "rating": str(review['rating']),
                    "reviewer": review['user_name'],
                    "source": "review"
                }
            ))
        except Exception as e:
            print(f"⚠️ Error creating review document: {e}")
            continue
    
    print(f"✅ Created {len(documents)} documents total")
    return documents

def vectorize_and_upload(documents, embeddings, index):
    """Vectorize documents and upload to Pinecone"""
    print(f"🔄 Starting vectorization of {len(documents)} documents...")
    
    # Clear existing vectors
    try:
        print("🧹 Clearing existing vectors from index...")
        index.delete(delete_all=True)
        time.sleep(10)
        print("✅ Index cleared")
    except Exception as e:
        print(f"⚠️ Note: Could not clear index (might be empty): {e}")
    
    vectors_to_upsert = []
    batch_size = 50
    successful_uploads = 0
    
    for i, doc in enumerate(documents):
        try:
            # Generate embedding
            embedding = embeddings.embed_query(doc.page_content)
            
            # Ensure it's a list of floats
            if not isinstance(embedding, list):
                embedding = list(embedding)
            embedding = [float(x) for x in embedding]
            
            # Verify embedding dimension
            if len(embedding) != 384:
                print(f"⚠️ WARNING: Embedding dimension {len(embedding)} != 384")
                continue
            
            # Create unique vector ID
            vector_id = f"doc_{i}_{uuid.uuid4().hex[:8]}"
            
            # Prepare metadata
            metadata = {
                "text": doc.page_content[:1000],
                "type": doc.metadata.get("type", "unknown"),
                "product_id": doc.metadata.get("product_id", ""),
                "name": doc.metadata.get("name", "")[:200],
                "category": doc.metadata.get("category", "")[:200],
                "source": doc.metadata.get("source", "unknown")
            }
            
            # Add review-specific metadata
            if doc.metadata.get("type") == "review":
                metadata["rating"] = doc.metadata.get("rating", "")
                metadata["reviewer"] = doc.metadata.get("reviewer", "")[:200]
            
            vector_record = {
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }
            vectors_to_upsert.append(vector_record)
            
            # Upsert in batches
            if len(vectors_to_upsert) >= batch_size:
                print(f"📤 Upserting batch {(i // batch_size) + 1} with {len(vectors_to_upsert)} vectors...")
                try:
                    index.upsert(vectors=vectors_to_upsert)
                    successful_uploads += len(vectors_to_upsert)
                    vectors_to_upsert = []
                    time.sleep(2)  # Small delay between batches
                except Exception as e:
                    print(f"❌ ERROR in batch upsert: {e}")
                    vectors_to_upsert = []
            
            if (i + 1) % 20 == 0:
                print(f"📊 Processed {i + 1}/{len(documents)} documents")
        
        except Exception as e:
            print(f"❌ ERROR processing document {i}: {e}")
            continue
    
    # Upsert any remaining vectors
    if vectors_to_upsert:
        print(f"📤 Upserting final batch with {len(vectors_to_upsert)} vectors...")
        try:
            index.upsert(vectors=vectors_to_upsert)
            successful_uploads += len(vectors_to_upsert)
        except Exception as e:
            print(f"❌ ERROR in final upsert: {e}")
    
    print(f"✅ Vectorization complete! Successfully uploaded {successful_uploads} vectors")
    return successful_uploads

def verify_upload(index):
    """Verify that vectors were uploaded"""
    try:
        stats = index.describe_index_stats()
        print(f"📊 Pinecone Index Stats:")
        print(f"   Total Vectors: {stats.total_vector_count}")
        print(f"   Dimension: {stats.dimension}")
        
        return stats.total_vector_count > 0
    except Exception as e:
        print(f"❌ ERROR verifying upload: {e}")
        return False

def main():
    """Main vectorization workflow"""
    print("=" * 60)
    print("🚀 Starting Pinecone Vectorization Process")
    print("=" * 60)
    
    try:
        # Initialize components
        embeddings = init_embeddings()
        index = init_pinecone()
        
        # Fetch data
        print("🔄 Fetching data from MySQL database...")
        products = fetch_products()
        reviews = fetch_reviews()
        
        if not products and not reviews:
            print("❌ ERROR: No data found in database")
            sys.exit(1)
        
        # Create documents
        documents = create_documents(products, reviews)
        
        if not documents:
            print("❌ ERROR: No documents created")
            sys.exit(1)
        
        # Vectorize and upload
        uploaded_count = vectorize_and_upload(documents, embeddings, index)
        
        if uploaded_count == 0:
            print("❌ ERROR: No vectors were uploaded")
            sys.exit(1)
        
        # Verify
        print("🔍 Verifying upload...")
        success = verify_upload(index)
        
        if success:
            print("\n🎉 Vectorization completed successfully!")
            print(f"✅ Uploaded {uploaded_count} vectors to Pinecone")
            print("🤖 Your Pinecone vector store is now populated and ready for AI queries.")
        else:
            print("\n⚠️ Vectorization may have issues. Check the logs above.")
    
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
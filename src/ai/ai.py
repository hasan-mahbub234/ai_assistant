# ai_customer_support.py
import os
from dotenv import load_dotenv
load_dotenv()

import pymysql
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import requests
import langdetect
from langdetect import detect, LangDetectException
import uuid
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_pinecone_instance = None
_pinecone_lock = None

# Text cleaning functions
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

class GroqChat:
    def __init__(self, api_key, model_name="llama-3.1-8b-instant", temperature=0.1):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def invoke(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            return "I apologize, but I'm having trouble processing your request right now."

class LanguageDetector:
    """Detect user language (Bengali or English)"""
    
    @staticmethod
    def detect_language(text):
        """Detect if text is in Bengali or English"""
        try:
            lang = detect(text)
            if lang == 'bn':
                return 'bengali'
            elif lang == 'en':
                return 'english'
            else:
                return 'english'
        except LangDetectException:
            return 'english'

class MultilingualPromptBuilder:
    """Build prompts in appropriate language"""
    
    ENGLISH_SYSTEM_PROMPT = """You are an AI customer support assistant for an ecommerce website. 
    Use the provided context to answer the user's question accurately and helpfully.
    
    Your responsibilities:
    1. Provide product descriptions and information
    2. Answer questions about product reviews and quality
    3. Help customers create and manage orders
    4. Assist with returns, shipping, and policies
    
    If the information isn't in the context, politely say you don't know and suggest 
    contacting customer support for more specific queries."""
    
    BENGALI_SYSTEM_PROMPT = """আপনি একটি ই-কমার্স ওয়েবসাইটের জন্য একজন AI গ্রাহক সহায়তা সহায়ক।
    ব্যবহারকারীর প্রশ্নের উত্তর দিতে প্রদত্ত প্রসঙ্গ ব্যবহার করুন এবং সঠিক ও সহায়ক উত্তর দিন।
    
    আপনার দায়িত্ব:
    1. পণ্যের বিবরণ এবং তথ্য প্রদান করা
    2. পণ্যের পর্যালোচনা এবং গুণমান সম্পর্কে প্রশ্নের উত্তর দেওয়া
    3. গ্রাহকদের অর্ডার তৈরি এবং পরিচালনায় সহায়তা করা
    4. রিটার্ন, শিপিং এবং নীতি সম্পর্কে সহায়তা করা
    
    যদি প্রসঙ্গে তথ্য না থাকে, বিনয়ের সাথে বলুন আপনি জানেন না এবং গ্রাহক সহায়তার সাথে যোগাযোগ করার পরামর্শ দিন।"""
    
    @staticmethod
    def build_prompt(user_query, context, language='english', user_context=None):
        """Build language-specific prompt"""
        
        if language == 'bengali':
            system_prompt = MultilingualPromptBuilder.BENGALI_SYSTEM_PROMPT
            user_content = f"""আমাদের জ্ঞান ভিত্তি থেকে প্রাসঙ্গিক তথ্য:
{context}

ব্যবহারকারীর প্রশ্ন: {user_query}

উপরের তথ্যের উপর ভিত্তি করে একটি সহায়ক প্রতিক্রিয়া প্রদান করুন।"""
        else:
            system_prompt = MultilingualPromptBuilder.ENGLISH_SYSTEM_PROMPT
            user_content = f"""Relevant Information from our knowledge base:
{context}

User Question: {user_query}

Please provide a helpful response based on the information above."""
        
        return system_prompt, user_content

class EcommerceAICustomerSupport:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EcommerceAICustomerSupport, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if EcommerceAICustomerSupport._initialized:
            return
        
        logger.info("🚀 Initializing EcommerceAICustomerSupport (first and only time)")
        
        # Initialize Groq LLM
        self.llm = GroqChat(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.1
        )
        
        self.language_detector = LanguageDetector()
        self.prompt_builder = MultilingualPromptBuilder()
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize database connection
        self.db_config = {
            'host': os.getenv('DB_HOST', '43.204.142.93'),
            'user': os.getenv('DB_USER', 'mahbubdb'),
            'password': os.getenv('DB_PASSWORD', 'mahbub123'),
            'database': os.getenv('DB_NAME', 'ecommerce_website'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'charset': 'utf8mb4'
        }
        
        # Simple conversation memory
        self.conversation_history = []
        
        self.pinecone_available = self.setup_pinecone()
        
        self.auto_vectorize_on_startup()
        
        EcommerceAICustomerSupport._initialized = True
        
        if self.pinecone_available:
            logger.info("✅ Pinecone initialized successfully (singleton)")
        else:
            logger.warning("⚠️ Pinecone not available - AI will use database queries only")

    def setup_pinecone(self):
        if hasattr(self, 'index') and self.index is not None:
            logger.info("✅ Pinecone already initialized, skipping setup")
            return True
        
        """Create or connect to Pinecone index"""
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                logger.error("❌ PINECONE_API_KEY not found")
                self.index = None
                return False
            
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index_name = os.getenv("PINECONE_INDEX_NAME", "ecommerce-customer-support")
            
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [index.name for index in existing_indexes.indexes]
            
            if self.index_name not in index_names:
                logger.warning(f"❌ Pinecone index '{self.index_name}' not found")
                logger.info("💡 Index will be created and populated on startup")
                self.index = None
                return False
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"✅ Connected to Pinecone index: {self.index_name}")
            
            # Test the connection
            stats = self.index.describe_index_stats()
            logger.info(f"📊 Pinecone has {stats.total_vector_count} vectors")
            
            return True
                
        except Exception as e:
            logger.error(f"❌ Error initializing Pinecone: {e}")
            self.index = None
            return False

    def auto_vectorize_on_startup(self):
        """Automatically vectorize database on startup if Pinecone is empty"""
        try:
            if not hasattr(self, 'index') or self.index is None:
                logger.info("📦 Skipping auto-vectorization: Pinecone index not available")
                return
            
            stats = self.index.describe_index_stats()
            vector_count = stats.total_vector_count
            
            if vector_count == 0:
                logger.info("🔄 Pinecone is empty. Starting auto-vectorization...")
                self.vectorize_database()
            else:
                logger.info(f"✅ Pinecone already has {vector_count} vectors. Skipping vectorization.")
        
        except Exception as e:
            logger.warning(f"⚠️ Error checking vectorization status: {e}")

    def vectorize_database(self):
        """Vectorize products and reviews from database"""
        try:
            logger.info("🚀 Starting automatic database vectorization...")
            
            connection = self.get_db_connection()
            products = []
            reviews = []
            
            # Fetch products
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("""
                    SELECT uid, name, description, price, brand, category, quantity
                    FROM Products LIMIT 100
                """)
                products = cursor.fetchall()
                logger.info(f"✅ Fetched {len(products)} products")
                
                # Fetch reviews
                cursor.execute("""
                    SELECT rating, review_text, product_uid, user_name
                    FROM ProductReview LIMIT 500
                """)
                reviews = cursor.fetchall()
                logger.info(f"✅ Fetched {len(reviews)} reviews")
            
            connection.close()
            
            # Create documents and vectorize
            vectors_to_upsert = []
            doc_count = 0
            
            # Process products
            for i, product in enumerate(products):
                try:
                    product_id = str(uuid.UUID(bytes=product['uid'])) if isinstance(product['uid'], bytes) else str(product['uid'])
                    clean_description = extract_clean_text(product.get('description', ''), 200)
                    
                    product_text = f"""Product: {product['name']}
Brand: {product.get('brand', 'Not specified')}
Category: {product.get('category', 'Not specified')}
Price: ${product.get('price', 0)}
Description: {clean_description}
Available Quantity: {product.get('quantity', 0)} units"""
                    
                    embedding = self.embeddings.embed_query(product_text)
                    embedding = [float(x) for x in embedding]
                    
                    vectors_to_upsert.append({
                        "id": f"product_{i}_{uuid.uuid4().hex[:8]}",
                        "values": embedding,
                        "metadata": {
                            "text": product_text[:1000],
                            "type": "product",
                            "product_id": product_id,
                            "name": product['name'][:200],
                            "source": "product"
                        }
                    })
                    doc_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error processing product {i}: {e}")
            
            # Process reviews
            for i, review in enumerate(reviews):
                try:
                    clean_review = clean_html(review.get('review_text', ''))
                    review_text = f"Review Rating: {review['rating']}/5\nReviewer: {review['user_name']}\nComment: {clean_review}"
                    
                    embedding = self.embeddings.embed_query(review_text)
                    embedding = [float(x) for x in embedding]
                    
                    vectors_to_upsert.append({
                        "id": f"review_{i}_{uuid.uuid4().hex[:8]}",
                        "values": embedding,
                        "metadata": {
                            "text": review_text[:1000],
                            "type": "review",
                            "rating": str(review['rating']),
                            "reviewer": review['user_name'][:200],
                            "source": "review"
                        }
                    })
                    doc_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error processing review {i}: {e}")
            
            # Upsert all vectors
            if vectors_to_upsert:
                logger.info(f"📤 Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
                self.index.upsert(vectors=vectors_to_upsert)
                logger.info(f"✅ Auto-vectorization complete! {len(vectors_to_upsert)} vectors uploaded")
            
        except Exception as e:
            logger.error(f"❌ Error during auto-vectorization: {e}")

    def generate_chat_id(self, user_id=None):
        """Generate unique chat ID for conversation"""
        if user_id:
            chat_id = f"user_{user_id}_{uuid.uuid4().hex[:8]}"
        else:
            chat_id = f"guest_{uuid.uuid4().hex[:12]}"
        
        return chat_id

    def search_similar_documents(self, query, k=3):
        """Search for similar documents in Pinecone"""
        try:
            if not hasattr(self, 'index') or self.index is None:
                return []
            
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            documents = []
            for match in results.matches:
                # Clean the text before creating document
                clean_content = clean_html(match.metadata.get("text", ""))
                doc = Document(
                    page_content=clean_content,
                    metadata=match.metadata
                )
                documents.append(doc)
            
            logger.info(f"✅ Found {len(documents)} documents via Pinecone")
            return documents
                
        except Exception as e:
            logger.error(f"❌ Error searching Pinecone: {e}")
            return []

    def get_db_connection(self):
        """Create MySQL database connection"""
        return pymysql.connect(**self.db_config)

    def search_products_from_db(self, keyword):
        """Search products directly from database by name or keyword"""
        connection = self.get_db_connection()
        products = []
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                # Search products by name, brand, or description
                query = """
                    SELECT uid, name, description, price, brand, category, quantity 
                    FROM Products 
                    WHERE name LIKE %s OR brand LIKE %s OR category LIKE %s OR description LIKE %s
                    LIMIT 10
                """
                search_term = f"%{keyword}%"
                cursor.execute(query, (search_term, search_term, search_term, search_term))
                raw_results = cursor.fetchall()
                
                for row in raw_results:
                    # Clean the description
                    clean_description = extract_clean_text(row['description'], 200)
                    
                    products.append({
                        'uid': str(uuid.UUID(bytes=row['uid'])) if isinstance(row['uid'], bytes) else str(row['uid']),
                        'name': row['name'],
                        'description': clean_description,
                        'price': row['price'],
                        'brand': row['brand'],
                        'category': row['category'],
                        'quantity': row['quantity']
                    })
                    
        except Exception as e:
            print(f"❌ Error searching products from database: {e}")
        finally:
            connection.close()
        
        return products

    def get_product_recommendations_with_reviews(self, keyword, limit=5):
        """Search for products and include their reviews for recommendations"""
        connection = self.get_db_connection()
        recommendations = []
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                # Search for products
                query = """
                    SELECT uid, name, description, price, brand, category, quantity,
                           (SELECT AVG(rating) FROM ProductReview WHERE product_uid = Products.uid) as avg_rating,
                           (SELECT COUNT(*) FROM ProductReview WHERE product_uid = Products.uid) as review_count
                    FROM Products 
                    WHERE name LIKE %s OR brand LIKE %s OR category LIKE %s OR description LIKE %s
                    ORDER BY review_count DESC, avg_rating DESC
                    LIMIT %s
                """
                search_term = f"%{keyword}%"
                cursor.execute(query, (search_term, search_term, search_term, search_term, limit))
                raw_results = cursor.fetchall()
                
                for row in raw_results:
                    product_uid_str = str(uuid.UUID(bytes=row['uid'])) if isinstance(row['uid'], bytes) else str(row['uid'])
                    
                    # Get top 3 reviews for this product
                    reviews_query = """
                        SELECT rating, review_text, user_name
                        FROM ProductReview 
                        WHERE product_uid = %s
                        ORDER BY rating DESC, created_at DESC
                        LIMIT 3
                    """
                    cursor.execute(reviews_query, (row['uid'],))
                    reviews = cursor.fetchall()
                    
                    # Clean the description
                    clean_description = extract_clean_text(row['description'], 150)
                    
                    # Clean and format reviews
                    clean_reviews = []
                    for r in reviews:
                        clean_review_text = clean_html(r['review_text'])
                        clean_reviews.append({
                            'rating': r['rating'],
                            'text': clean_review_text[:100],
                            'reviewer': r['user_name']
                        })
                    
                    recommendations.append({
                        'product_id': product_uid_str,
                        'name': row['name'],
                        'description': clean_description,
                        'price': row['price'],
                        'brand': row['brand'],
                        'category': row['category'],
                        'available_quantity': row['quantity'],
                        'average_rating': float(row['avg_rating']) if row['avg_rating'] else 0,
                        'total_reviews': row['review_count'] or 0,
                        'top_reviews': clean_reviews
                    })
                    
        except Exception as e:
            print(f"❌ Error getting product recommendations: {e}")
        finally:
            connection.close()
        
        return recommendations

    def get_product_reviews_from_db(self, product_name):
        """Get all reviews for a specific product by name"""
        connection = self.get_db_connection()
        reviews = []
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                # First find the product by name
                product_query = """
                    SELECT uid FROM Products WHERE name LIKE %s LIMIT 1
                """
                cursor.execute(product_query, (f"%{product_name}%",))
                product = cursor.fetchone()
                
                if product:
                    product_uid = product['uid']
                    
                    # Get all reviews for this product
                    reviews_query = """
                        SELECT rating, review_text, user_name, created_at
                        FROM ProductReview 
                        WHERE product_uid = %s
                        ORDER BY rating DESC, created_at DESC
                    """
                    cursor.execute(reviews_query, (product_uid,))
                    raw_reviews = cursor.fetchall()
                    
                    for review in raw_reviews:
                        clean_review = clean_html(review['review_text'])
                        reviews.append({
                            'rating': review['rating'],
                            'text': clean_review,
                            'reviewer': review['user_name'],
                            'date': str(review['created_at'])[:10] if review['created_at'] else 'Unknown date'
                        })
                    
                    print(f"✅ Found {len(reviews)} reviews for product: {product_name}")
                else:
                    print(f"❌ Product not found: {product_name}")
                    
        except Exception as e:
            print(f"❌ Error getting product reviews: {e}")
        finally:
            connection.close()
        
        return reviews

    def extract_relevant_context(self, query, context):
        """Filter and show only relevant information based on question"""
        try:
            query_lower = query.lower()
            
            # Extract relevant keywords from query
            review_keywords = ['review', 'reviews', 'rating', 'ratings', 'feedback', 'opinion', 'মতামত', 'রেটিং']
            price_keywords = ['price', 'cost', 'expensive', 'cheap', 'budget', 'afford', 'দাম', 'মূল্য']
            availability_keywords = ['available', 'stock', 'quantity', 'buy', 'purchase', 'স্টক', 'উপলব্ধ']
            description_keywords = ['feature', 'features', 'specification', 'specs', 'what', 'product', 'about', 'বৈশিষ্ট্য', 'পণ্য']
            
            # Determine query type
            is_review = any(kw in query_lower for kw in review_keywords)
            is_price = any(kw in query_lower for kw in price_keywords)
            is_availability = any(kw in query_lower for kw in availability_keywords)
            is_description = any(kw in query_lower for kw in description_keywords)
            
            # Filter context based on query type
            filtered_lines = []
            for line in context.split('\n'):
                line_lower = line.lower()
                
                # Include relevant lines
                if is_review and any(kw in line_lower for kw in ['review', 'rating', 'feedback', 'রেটিং', 'মতামত']):
                    filtered_lines.append(line)
                elif is_price and any(kw in line_lower for kw in ['price', 'cost', 'afford', 'দাম']):
                    filtered_lines.append(line)
                elif is_availability and any(kw in line_lower for kw in ['quantity', 'available', 'stock', 'স্টক']):
                    filtered_lines.append(line)
                elif is_description and any(kw in line_lower for kw in ['description', 'feature', 'specification', 'বৈশিষ্ট্য']):
                    filtered_lines.append(line)
                # Always include important headers and summaries
                elif any(header in line_lower for header in ['product:', 'brand:', 'category:', 'summary:']):
                    filtered_lines.append(line)
            
            return '\n'.join(filtered_lines) if filtered_lines else context
        
        except Exception as e:
            logger.warning(f"⚠️ Error filtering context: {e}")
            return context

    def get_customer_response(self, user_query, user_context=None, user_id=None, chat_id=None, conversation_history=None, product_context=None):
        """Get AI response with both Pinecone and database integration, with conversation context"""
        try:
            if not chat_id:
                chat_id = self.generate_chat_id(user_id)
        
            detected_language = self.language_detector.detect_language(user_query)
            logger.info(f"💬 Chat ID: {chat_id} | User: {user_id or 'guest'} | Language: {detected_language}")
            logger.info(f"📝 Query: {user_query}")
        
            context = ""
            pinecone_docs = []
            tracked_products = product_context or {}
        
            if conversation_history and len(conversation_history) > 0:
                logger.info(f"[v0] Conversation history found: {len(conversation_history)} messages")
                last_user_messages = [msg for msg in conversation_history if msg.get("role") == "user"][-2:] if conversation_history else []
                logger.info(f"[v0] Last user messages: {[m.get('content')[:30] for m in last_user_messages]}")
        
            # First, try Pinecone vector search
            if self.pinecone_available:
                logger.info("🔍 Searching Pinecone vector store...")
                pinecone_docs = self.search_similar_documents(user_query, k=3)
                if pinecone_docs:
                    context += "Relevant Information:\n"
                    for doc in pinecone_docs:
                        context += f"- {doc.page_content}\n\n"
                    logger.info("✅ Using Pinecone context")
        
            # Check for specific product review queries
            review_keywords = ['review', 'reviews', 'rating', 'ratings', 'feedback', 'opinion', 'মতামত', 'রেটিং', 'পর্যালোচনা']
            is_review_query = any(keyword in user_query.lower() for keyword in review_keywords)
        
            # Extract product name from query or use from conversation context
            product_name = None
        
            if tracked_products:
                for prod_name in tracked_products.keys():
                    if any(word in user_query.lower() for word in prod_name.lower().split()):
                        product_name = prod_name
                        logger.info(f"[v0] Recognized product from conversation: {product_name}")
                        break
        
            # If no tracked product found, extract from current query
            if not product_name and is_review_query:
                words = user_query.split()
                for i, word in enumerate(words):
                    if word.lower() in ['product', 'item', 'about', 'for', 'পণ্য', 'আইটেম'] and i + 1 < len(words):
                        product_name = ' '.join(words[i+1:i+4])
                        break
            
                if not product_name:
                    common_words = ['tell', 'me', 'about', 'the', 'product', 'reviews', 'review', 'people', 'what', 'are', 'is', 'of']
                    product_words = [word for word in words if word.lower() not in common_words and len(word) > 2]
                    if product_words:
                        product_name = ' '.join(product_words[:3])
        
            # If it's a review query and we have a product name, get specific reviews
            if is_review_query and product_name:
                logger.info(f"🔍 Searching for reviews of: {product_name}")
                product_reviews = self.get_product_reviews_from_db(product_name)
            
                if product_reviews:
                    context += f"\nCustomer Reviews for '{product_name}':\n"
                    for i, review in enumerate(product_reviews[:5]):
                        stars = "⭐" * review['rating']
                        context += f"\n{i+1}. {stars} ({review['rating']}/5) by {review['reviewer']}\n"
                        context += f"   \"{review['text']}\"\n"
                        context += f"   Date: {review['date']}\n"
                
                    total_reviews = len(product_reviews)
                    avg_rating = sum(r['rating'] for r in product_reviews) / total_reviews if total_reviews > 0 else 0
                    context += f"\nSummary: {total_reviews} total reviews, Average rating: {avg_rating:.1f}/5\n"
                    logger.info(f"✅ Found {len(product_reviews)} reviews")
                
                    tracked_products[product_name] = {
                        "avg_rating": avg_rating,
                        "total_reviews": total_reviews,
                        "mentioned_at": str(__import__('datetime').datetime.now())
                    }
        
            # If no specific context found yet, use general product search
            if not context or (not pinecone_docs and not is_review_query):
                logger.info("🔍 Using general product search...")
            
                words = user_query.split()
                search_keywords = [w for w in words if len(w) > 2]
            
                for keyword in search_keywords:
                    recommendations = self.get_product_recommendations_with_reviews(keyword, limit=2)
                
                    if recommendations:
                        logger.info(f"✅ Found {len(recommendations)} products in database")
                        context += "\nProduct Information:\n"
                        for product in recommendations:
                            stars = "⭐" * int(product['average_rating'])
                            context += f"\n- Product: {product['name']}"
                            context += f"\n  Brand: {product['brand']}"
                            context += f"\n  Price: ${product['price']}"
                            context += f"\n  Category: {product['category']}"
                            context += f"\n  Description: {product['description']}"
                            context += f"\n  Rating: {stars} ({product['average_rating']}/5 from {product['total_reviews']} reviews)"
                        
                            if product['top_reviews']:
                                context += "\n  Recent Reviews:"
                                for review in product['top_reviews']:
                                    review_stars = "⭐" * review['rating']
                                    context += f"\n    - {review_stars} {review['reviewer']}: {review['text']}"
                        
                            tracked_products[product['name']] = {
                                "price": product['price'],
                                "avg_rating": product['average_rating'],
                                "total_reviews": product['total_reviews'],
                                "mentioned_at": str(__import__('datetime').datetime.now())
                            }
                        break
        
            context = self.extract_relevant_context(user_query, context)
        
            if not context:
                context = "No specific product information found. Please try rephrasing or ask about specific products."
        
            system_message, user_content = self.prompt_builder.build_prompt(
                user_query=user_query,
                context=context,
                language=detected_language
            )
        
            logger.info("🤖 Calling Groq API...")
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
        
            response = self.llm.invoke(messages)
            logger.info("✅ Groq response received")
        
            # Store conversation with chat_id
            self.conversation_history.append({
                "chat_id": chat_id,
                "user_id": user_id,
                "user": user_query,
                "assistant": response,
                "language": detected_language,
                "timestamp": str(__import__('datetime').datetime.now())
            })
        
            # Keep only last 50 messages
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-50:]
        
            return {
                "answer": response,
                "source_documents": pinecone_docs,
                "conversation_history": self.conversation_history,
                "language": detected_language,
                "chat_id": chat_id,
                "product_context": tracked_products  # Return tracked products
            }
        
        except Exception as e:
            logger.error(f"❌ Error in get_customer_response: {e}")
            import traceback
            traceback.print_exc()
        
            detected_language = self.language_detector.detect_language(user_query) if user_query else 'english'
            default_msg = "আমি ক্ষমাপ্রার্থী, কিন্তু আমি এখনই আপনার অনুরোধ প্রক্রিয়া করতে সমস্যা হচ্ছে। পরে আবার চেষ্টা করুন।" if detected_language == 'bengali' else "I apologize, but I'm having trouble processing your request. Please try again later."
        
            return {
                "answer": default_msg,
                "source_documents": [],
                "conversation_history": self.conversation_history,
                "language": detected_language,
                "chat_id": chat_id or "error",
                "product_context": product_context or {}  # Return product context even on error
            }

    def get_user_context(self, user_id=None):
        """Get user-specific context from database"""
        if not user_id:
            return None
        
        connection = self.get_db_connection()
        try:
            with connection.cursor() as cursor:
                # Check if orders table exists and get its structure
                cursor.execute("SHOW TABLES LIKE 'orders'")
                if not cursor.fetchone():
                    return None
                
                # Get user's recent orders
                cursor.execute("""SELECT id, status, total_amount FROM orders WHERE user_id = %s ORDER BY created_at DESC LIMIT 3""", (user_id,))
                recent_orders = cursor.fetchall()
                
                # Get user's cart items if cart_items table exists
                cursor.execute("SHOW TABLES LIKE 'cart_items'")
                if cursor.fetchone():
                    cursor.execute("""SELECT p.name, ci.quantity FROM cart_items ci JOIN products p ON ci.product_id = p.id WHERE ci.user_id = %s""", (user_id,))
                    cart_items = cursor.fetchall()
                else:
                    cart_items = []
                
        except Exception as e:
            print(f"Error getting user context: {e}")
            return None
        finally:
            connection.close()
        
        context = ""
        if recent_orders:
            context += f"User has {len(recent_orders)} recent orders. "
        if cart_items:
            context += f"User has {len(cart_items)} items in cart."
        
        return context if context else None

    def check_pinecone_status(self):
        """Check Pinecone status"""
        try:
            if not hasattr(self, 'index') or self.index is None:
                return {"status": "disconnected", "message": "Pinecone not initialized"}
            
            stats = self.index.describe_index_stats()
            return {
                "status": "connected", 
                "message": "Pinecone is working properly",
                "stats": {
                    "total_vector_count": stats.total_vector_count,
                    "dimension": stats.dimension
                }
            }
        except Exception as e:
            return {"status": "error", "message": f"Pinecone error: {str(e)}"}

# FastAPI Integration
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Ecommerce AI Customer Support",
    description="Multilingual AI Customer Support with Product Search & Order Tracking"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    sources: list
    success: bool = True

# Initialize AI support system
ai_support = EcommerceAICustomerSupport()

@app.get("/")
async def root():
    return {"message": "Ecommerce AI Customer Support API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    try:
        user_context = ai_support.get_user_context(request.user_id)
        
        result = ai_support.get_customer_response(
            user_query=request.message,
            user_context=user_context,
            user_id=request.user_id
        )
        
        # Extract source information
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            })
        
        return ChatResponse(
            response=result["answer"],
            sources=sources,
            success=True
        )
    
    except Exception as e:
        return ChatResponse(
            response="Sorry, I encountered an error. Please try again.",
            sources=[],
            success=False
        )




@app.get("/health")
async def health_check():
    """Health check endpoint"""
    pinecone_status = ai_support.check_pinecone_status()
    return {
        "status": "healthy", 
        "service": "AI Customer Support",
        "pinecone": pinecone_status

    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

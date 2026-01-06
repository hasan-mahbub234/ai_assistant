import logging
import uuid
from src.database import db

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Handle all database operations"""
    
    def search_products(self, keyword: str, limit: int = 10):
        """Search products by keyword"""
        connection = db.get_connection()
        products = []
        
        try:
            with connection.cursor() as cursor:
                query = """
                    SELECT p.uid, p.name, p.description, p.price, p.brand, 
                           p.category, p.quantity, p.discount, p.tags
                    FROM Products p
                    WHERE p.name LIKE %s OR p.brand LIKE %s OR p.category LIKE %s 
                          OR p.description LIKE %s OR p.tags LIKE %s
                    LIMIT %s
                """
                search_term = f"%{keyword}%"
                cursor.execute(query, (
                    search_term, search_term, search_term, 
                    search_term, search_term, limit
                ))
                
                for row in cursor.fetchall():
                    product_uid = row['uid']
                    if isinstance(product_uid, bytes):
                        product_id = str(uuid.UUID(bytes=product_uid))
                    else:
                        product_id = str(product_uid)
                    
                    products.append({
                        'product_id': product_id,
                        'name': row['name'],
                        'description': row['description'][:200] if row['description'] else '',
                        'price': float(row['price']) if row['price'] else 0.0,
                        'brand': row['brand'] or '',
                        'category': row['category'] or '',
                        'quantity': row['quantity'] or 0,
                        'discount': float(row['discount']) if row['discount'] else 0.0,
                        'tags': row['tags'] or ''
                    })
        except Exception as e:
            logger.error(f"Product search error: {e}")
        finally:
            connection.close()
        
        return products
    
    def get_product_with_reviews(self, keyword: str, limit: int = 5):
        """Get products with their review information"""
        connection = db.get_connection()
        recommendations = []
        
        try:
            with connection.cursor() as cursor:
                # Get products matching keyword
                query = """
                    SELECT p.uid, p.name, p.description, p.price, p.brand, 
                           p.category, p.quantity, p.discount,
                           COALESCE(AVG(pr.rating), 0) as avg_rating,
                           COUNT(pr.id) as review_count
                    FROM Products p
                    LEFT JOIN ProductReview pr ON p.uid = pr.product_uid
                    WHERE p.name LIKE %s OR p.brand LIKE %s OR p.category LIKE %s 
                          OR p.description LIKE %s OR p.tags LIKE %s
                    GROUP BY p.uid
                    ORDER BY review_count DESC, avg_rating DESC
                    LIMIT %s
                """
                search_term = f"%{keyword}%"
                cursor.execute(query, (
                    search_term, search_term, search_term, 
                    search_term, search_term, limit
                ))
                
                for row in cursor.fetchall():
                    product_uid = row['uid']
                    if isinstance(product_uid, bytes):
                        product_id = str(uuid.UUID(bytes=product_uid))
                    else:
                        product_id = str(product_uid)
                    
                    # Get top reviews for this product
                    cursor.execute("""
                        SELECT rating, review_text, user_name
                        FROM ProductReview 
                        WHERE product_uid = %s
                        ORDER BY created_at DESC LIMIT 3
                    """, (product_uid,))
                    
                    reviews = []
                    for review in cursor.fetchall():
                        reviews.append({
                            'rating': float(review['rating']) if review['rating'] else 0.0,
                            'text': (review['review_text'][:150] 
                                    if review['review_text'] else ''),
                            'reviewer': review['user_name'] or 'Anonymous'
                        })
                    
                    recommendations.append({
                        'product_id': product_id,
                        'name': row['name'],
                        'description': (row['description'][:150] 
                                       if row['description'] else ''),
                        'price': float(row['price']) if row['price'] else 0.0,
                        'brand': row['brand'] or '',
                        'category': row['category'] or '',
                        'quantity': row['quantity'] or 0,
                        'discount': float(row['discount']) if row['discount'] else 0.0,
                        'avg_rating': float(row['avg_rating']) or 0.0,
                        'review_count': int(row['review_count']) or 0,
                        'reviews': reviews
                    })
        except Exception as e:
            logger.error(f"Error getting products with reviews: {e}")
        finally:
            connection.close()
        
        return recommendations
    
    def get_order_info(self, order_id: str):
        """Get order tracking information"""
        connection = db.get_connection()
        
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT o.id, o.user_id, o.status, o.total_amount, 
                           o.payment_status, o.shipping_address, o.created_at
                    FROM orders o
                    WHERE o.id = %s
                """, (order_id,))
                
                order = cursor.fetchone()
                if not order:
                    return None
                
                # Get order items
                cursor.execute("""
                    SELECT p.name, oi.quantity, oi.price
                    FROM order_items oi
                    JOIN Products p ON oi.product_id = p.uid
                    WHERE oi.order_id = %s
                """, (order_id,))
                
                items = []
                for item in cursor.fetchall():
                    items.append({
                        'name': item['name'],
                        'quantity': item['quantity'],
                        'price': float(item['price']) if item['price'] else 0.0
                    })
                
                return {
                    'order_id': order['id'],
                    'user_id': order['user_id'],
                    'status': order['status'],
                    'total_amount': float(order['total_amount']) if order['total_amount'] else 0.0,
                    'payment_status': order['payment_status'],
                    'address': order['shipping_address'],
                    'created_at': str(order['created_at']),
                    'items': items
                }
        except Exception as e:
            logger.error(f"Error getting order info: {e}")
            return None
        finally:
            connection.close()
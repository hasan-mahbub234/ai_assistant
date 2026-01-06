import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build context from various sources for AI responses"""
    
    def build(self, products: List[Dict], query: str, 
              language: str = 'english') -> str:
        """Build complete context for AI"""
        if not products:
            return self._build_no_results_context(query, language)
        
        context_parts = []
        
        # Add product information
        product_context = self._format_products(products, language)
        context_parts.append(product_context)
        
        # Add query context
        query_context = self._format_query(query, language)
        context_parts.append(query_context)
        
        return "\n\n".join(context_parts)
    
    def _format_products(self, products: List[Dict], language: str) -> str:
        """Format product information"""
        if language == 'bengali':
            context = "প্রাপ্ত পণ্য সমূহ:\n\n"
            for i, product in enumerate(products, 1):
                context += f"{i}. {product.get('name', 'অজানা পণ্য')}\n"
                
                if product.get('brand'):
                    context += f"   ব্র্যান্ড: {product['brand']}\n"
                
                if product.get('price'):
                    context += f"   মূল্য: {product['price']:.2f} টাকা\n"
                
                if product.get('average_rating', 0) > 0:
                    rating = product['average_rating']
                    count = product.get('review_count', 0)
                    context += f"   রেটিং: {rating:.1f}/5 ({count} জনের রিভিউ)\n"
                
                if product.get('description'):
                    desc = product['description'][:120]
                    context += f"   বিবরণ: {desc}\n"
                
                if product.get('free_delivery'):
                    context += f"   বিনামূল্যে ডেলিভারি\n"
                
                context += "\n"
        else:
            context = "Available Products:\n\n"
            for i, product in enumerate(products, 1):
                context += f"{i}. {product.get('name', 'Unknown Product')}\n"
                
                if product.get('brand'):
                    context += f"   Brand: {product['brand']}\n"
                
                if product.get('price'):
                    context += f"   Price: {product['price']:.2f} BDT\n"
                
                if product.get('average_rating', 0) > 0:
                    rating = product['average_rating']
                    count = product.get('review_count', 0)
                    context += f"   Rating: {rating:.1f}/5 ({count} reviews)\n"
                
                if product.get('description'):
                    desc = product['description'][:120]
                    context += f"   Description: {desc}\n"
                
                if product.get('free_delivery'):
                    context += f"   Free Delivery Available\n"
                
                if product.get('best_selling'):
                    context += f"   Best Selling Product\n"
                
                context += "\n"
        
        return context.strip()
    
    def _format_query(self, query: str, language: str) -> str:
        """Format query for context"""
        if language == 'bengali':
            return f"ব্যবহারকারীর প্রশ্ন: {query}"
        else:
            return f"User Question: {query}"
    
    def _build_no_results_context(self, query: str, language: str) -> str:
        """Build context when no products found"""
        if language == 'bengali':
            return (
                f"ব্যবহারকারীর প্রশ্ন: {query}\n\n"
                "দ্রষ্টব্য: ডাটাবেসে এই নির্দিষ্ট পণ্য সম্পর্কে তথ্য পাওয়া যায়নি। "
                "আপনি অন্য কোনো পণ্য সম্পর্কে জানতে চান?"
            )
        else:
            return (
                f"User Question: {query}\n\n"
                "Note: No specific product information found in the database for this query. "
                "Would you like information about other products?"
            )
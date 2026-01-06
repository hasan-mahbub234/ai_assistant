from typing import List, Dict


class PromptBuilder:
    """Build prompts for different languages"""
    
    # System prompts for different languages
    SYSTEM_PROMPTS = {
        'english': """You are a helpful AI customer support assistant for an ecommerce website.

CRITICAL RULES:
1. Use ONLY plain text - NO markdown, NO tables, NO special formatting
2. Write in natural, conversational language
3. Be concise but complete
4. Always mention relevant products if available
5. Include price information when relevant
6. Mention customer ratings and reviews when available
7. If you don't have information, say so honestly

RESPONSE FORMAT:
- Plain text only
- No asterisks, no bold, no italics
- No markdown of any kind
- Simple line breaks for readability""",

        'bengali': """আপনি একটি ই-কমার্স ওয়েবসাইটের জন্য একজন সহায়ক AI গ্রাহক সেবা সহায়ক।

গুরুত্বপূর্ণ নিয়ম:
১. শুধুমাত্র সাধারণ পাঠ্য ব্যবহার করুন - কোন মার্কডাউন নেই, টেবিল নেই, বিশেষ ফর্ম্যাটিং নেই
২. প্রাকৃতিক, কথোপকথনমূলক ভাষায় লিখুন
৩. সংক্ষিপ্ত কিন্তু সম্পূর্ণ উত্তর দিন
৪. প্রাসঙ্গিক পণ্য থাকলে তাদের উল্লেখ করুন
৫. প্রাসঙ্গিক হলে মূল্য তথ্য অন্তর্ভুক্ত করুন
৬. গ্রাহক রেটিং এবং রিভিউ থাকলে উল্লেখ করুন
৭. তথ্য না থাকলে সৎভাবে বলুন

প্রতিক্রিয়া ফরম্যাট:
- শুধুমাত্র সাধারণ পাঠ্য
- কোন তারকাচিহ্ন নেই, বোল্ড নেই, ইটালিক নেই
- কোন মার্কডাউন নেই
- পড়ার সুবিধার জন্য সহজ লাইন ব্রেক""",

        'banglish': """You are a helpful AI customer support assistant for an ecommerce website.

Important rules:
1. Use simple text only - no special formatting
2. Write in Banglish (Bengali in Latin script) like a friendly shopkeeper
3. Be helpful and warm
4. Mention products with price and details
5. Use simple language everyone can understand
6. No markdown, no tables, just plain text

Example Banglish: "Ami apnake bolchi, ei product tar daam 500 taka. Customer ra bolechen valo product."""
}
    
    def build(self, query: str, context: str, language: str = 'english') -> List[Dict]:
        """Build complete prompt for LLM"""
        system_prompt = self.SYSTEM_PROMPTS.get(language, self.SYSTEM_PROMPTS['english'])
        
        if "No specific product" in context or "ডাটাবেসে" in context:
            user_prompt = self._build_no_results_prompt(query, language)
        else:
            user_prompt = self._build_with_context_prompt(query, context, language)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _build_with_context_prompt(self, query: str, context: str, language: str) -> str:
        """Build prompt when context is available"""
        if language == 'bengali':
            return f"""প্রাসঙ্গিক তথ্য:
{context}

উপরের তথ্যের উপর ভিত্তি করে ব্যবহারকারীর প্রশ্নের উত্তর দিন:

ব্যবহারকারীর প্রশ্ন: {query}

দয়া করে:
- সরাসরি উত্তর দিন
- প্রাসঙ্গিক পণ্য উল্লেখ করুন
- মূল্য এবং অন্যান্য বিবরণ দিন
- সহজ বাংলায় লিখুন"""
        
        elif language == 'banglish':
            return f"""Available information:
{context}

Based on this information, answer the user's question:

User question: {query}

Please:
- Give direct answer
- Mention relevant products
- Include price and details
- Write in simple Banglish"""
        
        else:  # english
            return f"""Relevant Information:
{context}

Based on the information above, answer the user's question:

User Question: {query}

Please:
- Provide a direct answer
- Mention relevant products with details
- Include price information when applicable
- Write in simple, conversational English"""
    
    def _build_no_results_prompt(self, query: str, language: str) -> str:
        """Build prompt when no products found"""
        if language == 'bengali':
            return f"""ব্যবহারকারীর প্রশ্ন: {query}

দ্রষ্টব্য: ডাটাবেসে এই নির্দিষ্ট পণ্য সম্পর্কে তথ্য পাওয়া যায়নি।

দয়া করে ব্যবহারকারীকে বলুন যে আপনি এই পণ্য সম্পর্কে তথ্য খুঁজে পাচ্ছেন না এবং অন্য পণ্য সম্পর্কে সাহায্য করার প্রস্তাব দিন।"""
        
        elif language == 'banglish':
            return f"""User question: {query}

Note: No specific product information found in database.

Please tell the user you couldn't find information about this product and offer to help with other products."""
        
        else:  # english
            return f"""User Question: {query}

Note: No specific product information found in the database.

Please inform the user that you couldn't find information about this specific product and offer to help with other products."""
import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin, urlparse
import time
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import uvicorn
import os
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ScrapeRequest(BaseModel):
    website_url: str

class ErrorResponse(BaseModel):
    error: str
    status_code: int
    message: str

@dataclass
class Product:
    id: str
    title: str
    handle: str
    description: str
    vendor: str
    product_type: str
    created_at: str
    updated_at: str
    published_at: str
    tags: List[str]
    price: str
    compare_at_price: str
    available: bool
    url: str

@dataclass
class BrandData:
    store_name: str
    store_url: str
    scraped_at: str
    product_catalog: List[Product]
    hero_products: List[Dict]
    privacy_policy: str
    return_refund_policy: str
    faqs: List[Dict]
    social_handles: Dict[str, str]
    contact_details: Dict[str, Any]
    brand_context: str
    important_links: Dict[str, str]
    additional_insights: Dict[str, Any]

class ShopifyWebsiteScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.content_extractor = ContentExtractor()
        
    def get_page(self, url: str, timeout: int = 10) -> Optional[requests.Response]:
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.debug(f"Page not found (404): {url}")
            else:
                logger.error(f"HTTP error fetching {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def get_product_catalog(self) -> List[Product]:
        products = []
        page = 1
        max_pages = 5
        
        while page <= max_pages:
            url = f"{self.base_url}/products.json?limit=250&page={page}"
            response = self.get_page(url)
            
            if not response:
                logger.warning(f"Failed to fetch page {page}, stopping pagination")
                break
                
            try:
                data = response.json()
                if not data.get('products'):
                    logger.info(f"No products found on page {page}, stopping pagination")
                    break
                    
                page_products = data['products']
                logger.info(f"Found {len(page_products)} products on page {page}")
                    
                for product_data in page_products:
                    product = Product(
                        id=str(product_data.get('id', '')),
                        title=product_data.get('title', ''),
                        handle=product_data.get('handle', ''),
                        description=BeautifulSoup(product_data.get('body_html', ''), 'html.parser').get_text(strip=True),
                        vendor=product_data.get('vendor', ''),
                        product_type=product_data.get('product_type', ''),
                        created_at=product_data.get('created_at', ''),
                        updated_at=product_data.get('updated_at', ''),
                        published_at=product_data.get('published_at', ''),
                        tags=product_data.get('tags', []) if isinstance(product_data.get('tags'), list) else (product_data.get('tags', '').split(',') if product_data.get('tags') else []),
                        price=str(product_data.get('variants', [{}])[0].get('price', '')) if product_data.get('variants') else '',
                        compare_at_price=str(product_data.get('variants', [{}])[0].get('compare_at_price', '')) if product_data.get('variants') else '',
                        available=product_data.get('available', False),
                        url=f"{self.base_url}/products/{product_data.get('handle', '')}"
                    )
                    products.append(product)
                    
                if len(page_products) < 250:
                    logger.info(f"Page {page} returned fewer than 250 products, assuming end of catalog")
                    break
                    
                page += 1
                time.sleep(0.5)
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing products JSON for page {page}: {e}")
                break
                
        logger.info(f"Found {len(products)} total products across {page - 1} pages")
        return products
    
    def get_hero_products(self) -> List[Dict]:
        response = self.get_page(self.base_url)
        if not response:
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        hero_products = []
        
        selectors = [
            '.featured-product', '.hero-product', '.homepage-product',
            '.product-card', '.product-item', '[data-product-id]'
        ]
        
        for selector in selectors:
            products = soup.select(selector)
            for product in products:
                product_data = {}
                
                # Extract product title
                title_elem = product.select_one('.product-title, .product-name, h3, h4')
                if title_elem:
                    product_data['title'] = title_elem.get_text(strip=True)
                
                # Extract product link
                link_elem = product.select_one('a[href*="/products/"]')
                if link_elem:
                    product_data['url'] = urljoin(self.base_url, link_elem.get('href'))
                
                # Extract product image
                img_elem = product.select_one('img')
                if img_elem:
                    product_data['image'] = img_elem.get('src') or img_elem.get('data-src')
                
                # Extract price
                price_elem = product.select_one('.price, .product-price, [class*="price"]')
                if price_elem:
                    product_data['price'] = price_elem.get_text(strip=True)
                
                if product_data:
                    hero_products.append(product_data)
        
        return hero_products
    
    def get_privacy_policy(self) -> str:
        privacy_urls = [
            f"{self.base_url}/policies/privacy-policy",
            f"{self.base_url}/pages/privacy-policy", 
            f"{self.base_url}/privacy-policy",
            f"{self.base_url}/pages/privacy"
        ]
        
        for url in privacy_urls:
            response = self.get_page(url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                    element.decompose()
                
                content_selectors = ['.rte', '.policy-content', '.page-content', 'main', '.content']
                for selector in content_selectors:
                    content = soup.select_one(selector)
                    if content:
                        return content.get_text(strip=True)
                
                return soup.get_text(strip=True)
        
        return ""
    
    def get_return_refund_policy(self) -> str:
        policy_urls = [
            f"{self.base_url}/policies/refund-policy",
            f"{self.base_url}/pages/return-policy",
            f"{self.base_url}/pages/refund-policy",
            f"{self.base_url}/return-policy",
            f"{self.base_url}/refund-policy",
            f"{self.base_url}/pages/returns",
            f"{self.base_url}/pages/exchanges",
            f"{self.base_url}/pages/return-and-exchange-policy",
            f"{self.base_url}/policies/return-policy"
        ]
        
        for url in policy_urls:
            response = self.get_page(url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                    element.decompose()
                
                # Extract main content
                content_selectors = ['.rte', '.policy-content', '.page-content', 'main', '.content']
                for selector in content_selectors:
                    content = soup.select_one(selector)
                    if content:
                        return content.get_text(strip=True)
                
                return soup.get_text(strip=True)
        
        return ""
    
    async def get_faqs(self) -> List[Dict]:

        faqs = await self._get_faqs_traditional()
        
        unique_faqs = []
        seen_questions = set()
        
        for faq in faqs:
            if isinstance(faq, dict) and 'question' in faq:
                question_lower = faq['question'].lower().strip()
                if question_lower not in seen_questions:
                    seen_questions.add(question_lower)
                    unique_faqs.append(faq)
        
        return unique_faqs
    
    async def _discover_faq_urls(self) -> List[str]:
        
        faq_urls = []
        
        common_patterns = [
            f"{self.base_url}/pages/faqs",
            f"{self.base_url}/pages/faq",
            f"{self.base_url}/faq",
            f"{self.base_url}/help",
            f"{self.base_url}/pages/help",
            f"{self.base_url}/support",
            f"{self.base_url}/pages/support",
            f"{self.base_url}/pages/frequently-asked-questions"
        ]
        
        for url in common_patterns:
            response = self.get_page(url)
            if response and response.status_code == 200:
                faq_urls.append(url)
        
        response = self.get_page(self.base_url)
        if response:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                link_text = link.get_text(strip=True).lower()
                
                if href and 'faq' in href.lower():
                    full_url = urljoin(self.base_url, href)
                    if full_url not in faq_urls and full_url != self.base_url:
                        test_response = self.get_page(full_url)
                        if test_response and test_response.status_code == 200:
                            faq_urls.append(full_url)
                
                elif any(keyword in link_text for keyword in ['faq', 'frequently asked', 'questions', 'help', 'support']):
                    full_url = urljoin(self.base_url, href)
                    if full_url not in faq_urls and full_url != self.base_url:
                        test_response = self.get_page(full_url)
                        if test_response and test_response.status_code == 200:
                            faq_urls.append(full_url)
            
        return list(set(faq_urls))
    
    async def _get_faqs_traditional(self) -> List[Dict]:
        faqs = []
        
        faq_urls = await self._discover_faq_urls()
        
        for url in faq_urls:
            response = self.get_page(url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                    element.decompose()
                
                try:
                    content = soup.get_text()
                    if len(content) > 100:
                        extracted_faqs = await self.content_extractor.analyze_content(content, "faq")
                        if extracted_faqs and isinstance(extracted_faqs, list):
                            faqs.extend(extracted_faqs)
                            break
                except Exception as e:
                    logger.error(f"Content analysis failed for {url}: {e}")
        
        return faqs
    
    def get_social_handles(self) -> Dict[str, str]:
        """Extract social media handles"""
        response = self.get_page(self.base_url)
        if not response:
            return {}
            
        soup = BeautifulSoup(response.content, 'html.parser')
        social_handles = {}
        
        # Look for social media links
        social_patterns = {
            'instagram': r'instagram\.com/([^/\s]+)',
            'facebook': r'facebook\.com/([^/\s]+)',
            'twitter': r'twitter\.com/([^/\s]+)',
            'tiktok': r'tiktok\.com/@([^/\s]+)',
            'youtube': r'youtube\.com/([^/\s]+)',
            'linkedin': r'linkedin\.com/company/([^/\s]+)'
        }
        
        # Extract from links
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            for platform, pattern in social_patterns.items():
                match = re.search(pattern, href)
                if match:
                    social_handles[platform] = match.group(1)
        
        # Extract from text content
        text_content = soup.get_text()
        for platform, pattern in social_patterns.items():
            match = re.search(pattern, text_content)
            if match:
                social_handles[platform] = match.group(1)
        
        return social_handles
    
    def get_contact_details(self) -> Dict[str, Any]:
        contact_details = {}
        
        response = self.get_page(self.base_url)
        discovered_contact_urls = []
        
        if response:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                link_text = link.get_text(strip=True).lower()
                
                if href and any(keyword in href.lower() for keyword in ['contact', 'support']):
                    full_url = urljoin(self.base_url, href)
                    if full_url != self.base_url:
                        discovered_contact_urls.append(full_url)
                elif any(keyword in link_text for keyword in ['contact', 'support', 'get in touch']):
                    full_url = urljoin(self.base_url, href)
                    if full_url != self.base_url:
                        discovered_contact_urls.append(full_url)
        
        common_contact_urls = [
            f"{self.base_url}/pages/contact",
            f"{self.base_url}/contact",
            f"{self.base_url}/pages/contact-us"
        ]
        
        all_contact_urls = discovered_contact_urls + common_contact_urls
        
        contact_urls = []
        seen = set()
        for url in all_contact_urls:
            if url not in seen:
                contact_urls.append(url)
                seen.add(url)
        
        content = ""
        for url in contact_urls:
            response = self.get_page(url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                content = soup.get_text()
                break
        
        if not content:
            response = self.get_page(self.base_url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                footer = soup.select_one('footer')
                if footer:
                    content = footer.get_text()
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        if emails:
            contact_details['emails'] = list(set(emails))
        
        # Extract phone numbers
        phone_pattern = r'(\+?\d{1,4}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, content)
        if phones:
            contact_details['phones'] = list(set([''.join(phone) for phone in phones]))
        
        return contact_details
    
    async def get_brand_context(self) -> str:
        brand_context_urls = await self._discover_brand_context_urls()
        
        # Common about URLs as fallback
        common_about_urls = [
            f"{self.base_url}/pages/about",
            f"{self.base_url}/pages/about-us",
            f"{self.base_url}/about",
            f"{self.base_url}/pages/our-story",
            f"{self.base_url}/pages/mission",
            f"{self.base_url}/pages/vision",
            f"{self.base_url}/pages/story",
            f"{self.base_url}/story"
        ]
        
        # Prioritize discovered URLs, then add common patterns
        all_urls = brand_context_urls + common_about_urls
        
        # Remove duplicates while preserving order (discovered URLs first)
        about_urls = []
        seen = set()
        for url in all_urls:
            if url not in seen:
                about_urls.append(url)
                seen.add(url)
        
        # Extract content from each URL and analyze with AI
        for url in about_urls:
            response = self.get_page(url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                    element.decompose()
                
                # Extract main content
                raw_content = soup.get_text(strip=True)
                
                # Use AI to analyze and structure the content
                if raw_content and len(raw_content) > 100:  # Ensure meaningful content
                    try:
                        extracted_context = await self.content_extractor.analyze_content(raw_content, "brand_context")
                        if extracted_context and len(extracted_context) > 50:
                            logger.info(f"Extracted brand context from {url}")
                            return extracted_context
                    except Exception as e:
                        logger.error(f"Content analysis failed for {url}: {e}")
                        # Don't continue, try raw content fallback for this URL first
                        pass
                
                # Return raw content if AI fails but content exists
                if raw_content and len(raw_content) > 100:
                    logger.info(f"Using raw content from {url} as brand context")
                    return raw_content
        
        return ""
    
    async def _discover_brand_context_urls(self) -> List[str]:
        """Discover brand context URLs by searching for relevant links"""
        brand_urls = []
        
        response = self.get_page(self.base_url)
        if response:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for links that suggest brand context
            brand_keywords = ['about us', 'our story', 'about', 'story', 'mission', 'vision', 'values', 'history', 'founder', 'company']
            
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                link_text = link.get_text(strip=True).lower()
                
                # Check if link text suggests brand context
                if any(keyword in link_text for keyword in brand_keywords):
                    full_url = urljoin(self.base_url, href)
                    if full_url not in brand_urls and full_url != self.base_url:
                        # Verify the URL exists
                        test_response = self.get_page(full_url)
                        if test_response and test_response.status_code == 200:
                            brand_urls.append(full_url)
                
                # Check if href contains brand keywords as a fallback
                elif href and any(keyword in href.lower() for keyword in brand_keywords):
                    full_url = urljoin(self.base_url, href)
                    if full_url not in brand_urls and full_url != self.base_url:
                        # Verify the URL exists
                        test_response = self.get_page(full_url)
                        if test_response and test_response.status_code == 200:
                            brand_urls.append(full_url)
        
        return brand_urls
    
    def get_important_links(self) -> Dict[str, str]:
        """Extract important links from the website"""
        response = self.get_page(self.base_url)
        if not response:
            return {}
            
        soup = BeautifulSoup(response.content, 'html.parser')
        important_links = {}
        
        # Look for specific link patterns
        link_patterns = {
            'order_tracking': ['track', 'order', 'tracking'],
            'contact_us': ['contact', 'support'],
            'blog': ['blog', 'news', 'articles'],
            'shipping': ['shipping', 'delivery'],
            'size_guide': ['size', 'guide', 'chart'],
            'careers': ['careers', 'jobs'],
            'wholesale': ['wholesale', 'bulk']
        }
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            link_text = link.get_text(strip=True).lower()
            
            for category, keywords in link_patterns.items():
                if any(keyword in link_text for keyword in keywords):
                    important_links[category] = urljoin(self.base_url, href)
        
        return important_links
    
    def get_additional_insights(self) -> Dict[str, Any]:
        """Extract additional insights from the website"""
        insights = {}
        
        response = self.get_page(self.base_url)
        if not response:
            return insights
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract store name
        title_elem = soup.select_one('title')
        if title_elem:
            insights['store_title'] = title_elem.get_text(strip=True)
        
        # Extract meta description
        meta_desc = soup.select_one('meta[name="description"]')
        if meta_desc:
            insights['meta_description'] = meta_desc.get('content')
        
        # Check for currency
        currency_pattern = r'[£$€¥₹]'
        text_content = soup.get_text()
        currencies = re.findall(currency_pattern, text_content)
        if currencies:
            insights['detected_currencies'] = list(set(currencies))
        
        # Check for payment methods
        payment_methods = []
        payment_keywords = ['visa', 'mastercard', 'paypal', 'cod', 'cash on delivery', 'upi', 'razorpay']
        for keyword in payment_keywords:
            if keyword in text_content.lower():
                payment_methods.append(keyword)
        
        if payment_methods:
            insights['payment_methods'] = payment_methods
        
        return insights
    
    async def scrape_complete_data(self) -> BrandData:
        """Scrape all available data from the Shopify website"""
        logger.info(f"Starting comprehensive scraping of {self.base_url}")
        
        brand_data = BrandData(
            store_name="",
            store_url=self.base_url,
            scraped_at=datetime.now().isoformat(),
            product_catalog=[],
            hero_products=[],
            privacy_policy="",
            return_refund_policy="",
            faqs=[],
            social_handles={},
            contact_details={},
            brand_context="",
            important_links={},
            additional_insights={}
        )
        
        # Extract all data
        logger.info("Fetching product catalog...")
        brand_data.product_catalog = self.get_product_catalog()
        
        logger.info("Fetching hero products...")
        brand_data.hero_products = self.get_hero_products()
        
        logger.info("Fetching privacy policy...")
        brand_data.privacy_policy = self.get_privacy_policy()
        
        logger.info("Fetching return/refund policy...")
        brand_data.return_refund_policy = self.get_return_refund_policy()
        
        logger.info("Fetching FAQs...")
        brand_data.faqs = await self.get_faqs()
        
        logger.info("Fetching social handles...")
        brand_data.social_handles = self.get_social_handles()
        
        logger.info("Fetching contact details...")
        brand_data.contact_details = self.get_contact_details()
        
        logger.info("Fetching brand context...")
        brand_data.brand_context = await self.get_brand_context()
        
        logger.info("Fetching important links...")
        brand_data.important_links = self.get_important_links()
        
        logger.info("Fetching additional insights...")
        brand_data.additional_insights = self.get_additional_insights()
        
        # Set store name from insights
        if brand_data.additional_insights.get('store_title'):
            brand_data.store_name = brand_data.additional_insights['store_title']
        
        logger.info("Scraping completed successfully!")
        return brand_data
    
    def save_to_file(self, brand_data: BrandData, filename: str = None):
        """Save scraped data to JSON file"""
        if not filename:
            domain = urlparse(self.base_url).netloc.replace('.', '_')
            filename = f"shopify_data_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert dataclass to dict with custom serialization
        data_dict = asdict(brand_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved to {filename}")
        return filename

class ContentExtractor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    async def analyze_content(self, content: str, analysis_type: str) -> Any:
        """Generic content analysis."""
        try:
            if analysis_type == "faq":
                prompt = f"""
                Analyze the following content and extract FAQs in JSON format:
                
                {content}
                
                Return only valid JSON array with question-answer pairs:
                [
                  {{
                    "question": "Question text?",
                    "answer": "Answer text"
                  }}
                ]
                
                If no FAQs found, return: []
                """
            elif analysis_type == "brand_context":
                prompt = f"""
                Analyze the following content and extract brand context information:
                
                {content}
                
                Provide a comprehensive summary covering:
                - Brand story and mission
                - Company values and philosophy
                - Unique selling propositions
                - Target audience
                - Key differentiators
                
                Return only the brand context text, not in JSON format.
                """
            else:
                return None
            
            response = await self.model.generate_content_async(prompt)
            
            if analysis_type == "faq":
                try:
                    # Clean the response to extract JSON
                    content = response.text.strip()
                    if content.startswith('```json'):
                        content = content[7:-3]
                    elif content.startswith('```'):
                        content = content[3:-3]
                    
                    return json.loads(content)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse FAQ JSON from response")
                    return []
            else:
                return response.text.strip()
                
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return [] if analysis_type == "faq" else ""

@app.post("/scrape", tags=["Scraping"])
async def scrape_shopify_website(request: ScrapeRequest):
    try:
        website_url = request.website_url.strip()
        
        # Validate URL
        if not website_url:
            raise HTTPException(
                status_code=400,
                detail="Website URL is required"
            )
        
        # Add protocol if not present
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        
        # Validate URL format
        try:
            parsed_url = urlparse(website_url)
            if not parsed_url.netloc:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid URL format"
                )
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid URL format"
            )
        
        # Initialize scraper
        scraper = ShopifyWebsiteScraper(website_url)
        
        # Test if website is accessible
        test_response = scraper.get_page(website_url)
        if not test_response:
            raise HTTPException(
                status_code=401,
                detail="Website not found or not accessible"
            )
        
        # Check if it's likely a Shopify site
        try:
            products_test = scraper.get_page(f"{website_url}/products.json")
            if not products_test:
                logger.warning(f"Could not access /products.json for {website_url} - may not be a Shopify site")
        except Exception:
            logger.warning(f"Could not test Shopify endpoint for {website_url}")
        
        # Scrape all data
        logger.info(f"Starting scrape for {website_url}")
        brand_data = await scraper.scrape_complete_data()
        
        # Convert to dictionary for JSON response
        response_data = asdict(brand_data)
        
        # Add success metadata
        response_data["success"] = True
        response_data["scraped_at"] = datetime.now().isoformat()
        
        return JSONResponse(
            status_code=200,
            content=response_data
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while scraping {website_url}: {e}")
        raise HTTPException(
            status_code=401,
            detail="Website not found or not accessible"
        )
    except Exception as e:
        logger.error(f"Internal error while scraping {website_url}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during scraping"
        )

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }

if __name__ == "__main__":
    uvicorn.run("test:app", host="0.0.0.0", port=2000, reload=True)
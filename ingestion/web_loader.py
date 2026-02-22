import re
import time
import random
import requests
import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from .base_loader import Document, BaseLoader
from .chunker import Chunker

# Optional selenium imports for JavaScript-heavy sites
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

class WebLoader(BaseLoader):
    def __init__(self):
        super().__init__()
        self.chunker = Chunker(source_type="web")
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",

            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",

            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
            "Gecko/20100101 Firefox/121.0",

            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.0 Safari/605.1.15",
        ]
        
    def get_headers(self) -> dict:
        """Return realistic browser headers with a random user agent"""
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
    
    def fetch_with_requests(self, url: str):
        try:
            session = requests.Session()
            domain = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            session.get(domain, headers=self.get_headers(), timeout=10)

            time.sleep(random.uniform(1, 2))

            response = session.get(
                url,
                headers=self.get_headers(),
                timeout=15,
                allow_redirects=True
            )

            if response.status_code == 200:
                return response.text
            elif response.status_code == 403:
                print(f"[requests] 403 Forbidden — trying next strategy")
                return None
            else:
                print(f"[requests] Status {response.status_code}")
                return None

        except requests.RequestException as e:
            print(f"[requests] Failed: {e}")
            return None
        
    
    def fetch_with_selenium(self, url: str):            
        try:
            # Configure Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument(f"--user-agent={random.choice(self.user_agents)}")
            
            # Create driver
            driver = webdriver.Chrome(options=chrome_options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Navigate to page
            driver.get(url)
            
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            time.sleep(3)
            html = driver.page_source
            driver.quit()
            
            return html
            
        except Exception as e:
            print(f"[selenium] Failed: {e}")
            try:
                driver.quit()
            except:
                pass
            return None
        
        
    def fetch_html(self, url: str):
        print(f"\nFetching {url}")
        
        html = self.fetch_with_requests(url)
        if html:
            print("Fetched with requests")
            return html, "requests"

        html = self.fetch_with_selenium(url)
        if html:
            print("Fetched with selenium")
            return html, "selenium"
        return None, None
    
    
    def parse(self, html: str, url: str):
        soup = BeautifulSoup(html, 'html.parser')
        
        for tag in soup(["script", "style", "header", "nav", "footer", "aside", "form", "button"]):
            tag.decompose()
            
        main_content = (
            soup.find("article") or
            soup.find("main") or
            soup.find(id="content") or
            soup.find(class_="content") or
            soup.find("body")
        )
        
        text = main_content.get_text(separator="\n") if main_content else ""
        
        title = soup.find("title")
        title_text = title.get_text().strip() if title else urlparse(url).netloc
        
        return Document(
            content=text,
            metadata={
                "source": title_text,
                "url": url,
                "domain": urlparse(url).netloc,
                "doc_type": "web"
            }
        )
        
    
    def clean(self, doc: Document):
        text = doc.content

        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = [line for line in text.splitlines() if line.strip()]
        lines = [line for line in lines if len(line.split()) >= 3]
        text = "\n".join(lines)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'[^\x20-\x7E\n]', '', text)

        return Document(
            content=text.strip(),
            metadata=doc.metadata
        )
        
        
    def load(self, source: str):
        result = self.fetch_html(source)
        if not result or not result[0]:
            return []
        
        content, strategy = result
        raw_content = self.parse(content, source)
            
        if not raw_content:
            return []
        cleaned_content = self.clean(raw_content)
        chunks = self.chunker.chunk([cleaned_content])
        return chunks
    
        
    def load_urls(self, urls: list[str]):
        all_chunks = []
        failed_urls = []
        for url in urls:
            print(f"Loading URL: {url}")
            time.sleep(random.uniform(1, 3))
            chunks = self.load(url)
            if chunks:
                all_chunks.extend(chunks)
            else:
                failed_urls.append(url)
            
        return all_chunks, failed_urls

        
        

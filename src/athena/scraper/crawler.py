import requests
from bs4 import BeautifulSoup
import bz2
import io
import xml.etree.ElementTree as ET

def download_dump():
    """
    Download a compressed Wikipedia XML dump file and return a file-like BZ2 object for streaming.

    Returns:
        bz2.BZ2File: A file-like object representing the decompressed contents of the downloaded 
        Wikipedia XML dump. This object can be read line-by-line or by chunks as needed.

    Raises:
        requests.RequestException: If there is a network-related error during the file download.
    
    Note:
        The currently used URL points to a small (~92.3 MB) sample split of the full Wikipedia 
        dump to facilitate tests and development. For full-scale processing, replace the URL
        with the complete dump URL (~22.3 GB).
    """
    # url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2" # 22.3 GB
    url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles18.xml-p26716198p27121850.bz2" # 92.3 MB
    response = requests.get(url, stream=True, timeout=(10, 120))
    return bz2.BZ2File(io.BytesIO(response.content))

def crawl(dump, max_pages):
    """
    Parse a Wikipedia XML dump and extract the HTML content from a limited number of pages.

    Args:
        dump (file-like object): A file-like object representing the decompressed Wikipedia XML dump.
        max_pages (int): The maximum number of pages to process from the XML dump.

    Raises:
        xml.etree.ElementTree.ParseError: If the XML structure in the dump is malformed.
        AttributeError: If expected tags (e.g., 'title', 'revision', 'text') are missing in the XML.

    Notes:
        - The function uses a MediaWiki XML namespace map to locate relevant tags.
        - The text content of each page's revision is parsed using BeautifulSoup with the 
          'html.parser' backend for further processing.
        - If no text is found in a page's revision, that page is skipped.
    """
    tree = ET.parse(dump)
    root = tree.getroot()
    page_count = 0

    ns = {'mediawiki': 'http://www.mediawiki.org/xml/export-0.11/'}
    for page in root.findall("mediawiki:page", ns):
        if page_count >= max_pages:
            break
        #print(page)
        title = page.find("mediawiki:title", ns).text
        text = page.find("mediawiki:revision", ns).find("mediawiki:text", ns).text

        if text:
            content = BeautifulSoup(text, "html.parser")
            #print(title)
            #print(str(content))
            print("Full HTML content:", content.prettify())
            page_count += 1


def main():
    dump = download_dump()
    crawl(dump, 5)

if __name__ == "__main__":
    main()
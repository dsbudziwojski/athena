import requests
import mwparserfromhell
import bz2
import io
import json
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

def crawl(dump_object, max_pages):
    """
    Parse a Wikipedia XML dump and extract cleaned plain-text content from a limited number of pages.

    Args:
        dump_object (file-like object): A decompressed file-like object of the Wikipedia XML dump.
        max_pages (int): The maximum number of main namespace pages to extract and process.

    Returns:
        list of dict: A list where each dictionary represents a cleaned article with fields:
                      - 'id' (str): The page ID.
                      - 'title' (str): The article title.
                      - 'text' (str): The plain-text content with MediaWiki syntax removed.

    Raises:
        xml.etree.ElementTree.ParseError: If the XML structure in the dump is malformed.
        AttributeError: If expected tags (e.g., 'title', 'revision', 'text') are missing from a page element.

    Notes:
        - Only pages in the main namespace ('<ns>0</ns>') are included.
        - Redirect pages are skipped automatically based on the presence of a <redirect> tag.
        - Wikitext is parsed and cleaned using 'mwparserfromhell', and whitespace is stripped.
        - Cleaned articles are written to a file named 'dump_data.json' using UTF-8 encoding.
        - Pages with no text content or only minimal markup are excluded from the output.
    """
    tree = ET.parse(dump_object)
    root = tree.getroot()
    page_count = 0
    dump_data = []
    ns = {'mediawiki': 'http://www.mediawiki.org/xml/export-0.11/'}

    for page in root.findall("mediawiki:page", ns):
        if page_count >= max_pages:
            break
        if page.find("mediawiki:redirect", ns) is not None or page.find("mediawiki:ns", ns).text != "0":
            continue
        id = page.find("mediawiki:id", ns).text
        title = page.find("mediawiki:title", ns).text
        text = page.find("mediawiki:revision", ns).find("mediawiki:text", ns).text

        if text:
            processed_text = mwparserfromhell.parse(text).strip_code().strip()
            #print("\nPage Id: ", id, "\nPage Title: ", title, "\nPage Text: \n", processed_text)
            dump_data.append({
                "id": str(id),
                "title": str(title),
                "text": str(processed_text),
            })
            page_count += 1
    
    with open('dump_data.json', 'w', encoding='utf-8') as f:
        json.dump(dump_data, f, ensure_ascii=False, indent=4)

    return dump_data

def main():
    dump_object = download_dump()
    dump_data = crawl(dump_object, 25)
    print(dump_data)

if __name__ == "__main__":
    main()
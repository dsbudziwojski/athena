import requests
import mwparserfromhell
import bz2
import io
import json
import xml.etree.ElementTree as ET
from .utilities import get_data_path

class Scraper(object):
    """
    Initialize the Scraper object.

    Args:
        urls (list of str): List of URLs pointing to Wikipedia XML dump files.

    Attributes:
        self.urls (list of str): URLs to download dumps from.
        self.data_files (list of str): Paths to JSON files containing processed page data.
        self.dump_objects (list of bz2.BZ2File): List of decompressed dump file objects.
    """
    def __init__(self, urls):
        self.urls = urls
        self.data_files = []
        self.dump_objects = []
        self.crawl(25)

    def crawl(self, max_pages=1, dump_objects=None):
        """
        Extract plain-text content from a limited number of Wikipedia pages.

        Iterates through each decompressed dump object, parses XML to find main
        namespace pages (namespace 0), cleans the wikitext, and saves the result
        as JSON files in the training-data directory.

        Args:
            max_pages (int): Maximum number of pages to process per dump file.
            dump_objects (list of bz2.BZ2File): List of decompressed dump file objects.
        Raises:
            requests.RequestException: If there is an error during download.
            xml.etree.ElementTree.ParseError: If the XML structure in the dump is malformed.
            AttributeError: If expected tags (e.g., 'title', 'revision', 'text') are missing from a page element.

        Notes:
            - Only pages in the main namespace ('<ns>0</ns>') are included.
            - Redirect pages are skipped automatically based on the presence of a <redirect> tag.
            - Wikitext is parsed and cleaned using 'mwparserfromhell', and whitespace is stripped.
            - Cleaned articles are written to a file named 'dump_data.json' using UTF-8 encoding.
            - Pages with no text content or only minimal markup are excluded from the output.
        """
        if dump_objects is None:
            # url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2" # 22.3 GB
            # url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles18.xml-p26716198p27121850.bz2" # 92.3 MB
            for url in self.urls:
                response = requests.get(url, stream=True, timeout=(10, 120))
                self.dump_objects.append(bz2.BZ2File(io.BytesIO(response.content)))
        else:
            self.dump_objects = dump_objects

        i = 0
        for dump_object in self.dump_objects:
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

            path = get_data_path() / 'training-data'
            path.mkdir(parents=True, exist_ok=True)
            file_name = f'{i}.json'
            actual_path = path / file_name
            with open(actual_path, 'w', encoding='utf-8') as f:
                json.dump(dump_data, f, ensure_ascii=False, indent=4)
            self.data_files.append(str(actual_path))
            i += 1

    def get_dump_objects(self):
        """
        Returns:
            list of bz2.BZ2File: Decompressed dump file objects.
        """
        return self.dump_objects

    def get_data_files(self):
        """
        Returns:
            list of str: Paths to JSON files containing processed article data.
        """
        return self.data_files
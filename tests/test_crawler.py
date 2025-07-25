from athena.scraper.crawler import Scraper

print("Check if download_objects=None auto downloads the url for you")
s = Scraper(["https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles18.xml-p26716198p27121850.bz2"])
print(s.get_dump_objects())
print(s.get_data_files())
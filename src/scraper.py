from pathlib import Path
from hashlib import sha256
from bs4 import BeautifulSoup
from functools import lru_cache
import requests


@lru_cache(maxsize=1000)
def load_page(url):
    return requests.get(url)


def encode_str(text):
    return sha256(text.encode('utf-8')).hexdigest()


def dump_html(html, path):
    with open(path, encoding='utf-8', mode='w') as f:
        f.write(html)


def load_html(path):
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return f.read()
    else:
        return None

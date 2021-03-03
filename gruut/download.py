"""An example of wrapping manual tqdm updates for `urllib` reporthook.
See also: https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
"""

from urllib import request as urllib

from tqdm.auto import tqdm


def download_file(url: str, output_path: str, desc: str):
    """Downloads a file with progress shown"""
    response = urllib.urlopen(url)
    with tqdm.wrapattr(
        open(output_path, "wb"),
        "write",
        miniters=1,
        desc=desc,
        total=getattr(response, "length", None),
    ) as fout:
        for chunk in response:
            fout.write(chunk)

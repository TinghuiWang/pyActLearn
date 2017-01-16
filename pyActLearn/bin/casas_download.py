"""CASAS download script to get CASAS smart home datasets
"""

from __future__ import absolute_import
import os
import logging
import zipfile
import argparse
from fuel.downloaders.base import default_downloader

logger = logging.getLogger(__file__)

master_url = 'http://eecs.wsu.edu/~twang3/datasets/'
dataset_dict = {
    'B1': 'B1.zip',
    'B2': 'B2.zip',
    'B3': 'B3.zip',
    'bosch_legacy': 'bosch_legacy.zip'
}


def CASAS_download(directory, datasets):
    """Download CASAS datasets to directory

    Args:
        directory (:obj:`str`): path to directory to store the downloaded
        datasets (:obj:`tuple` of :obj:`str`): list of datasets to download
    """
    for dataset in datasets:
        filename = dataset_dict.get(dataset, None)
        if filename is None:
            print('Cannot find dataset %s' % dataset)
            print('Here are the available datasets:')
            for key in dataset_dict.keys():
                print('  * %s' % key)
        else:
            # Download zipped files
            default_downloader(directory=directory,
                               urls=[master_url + filename],
                               filenames=[filename],
                               clear=False)
            # Expand it in place
            file_path = os.path.join(directory, filename)
            if os.path.exists(file_path):
                zip_ref = zipfile.ZipFile(file_path, 'r')
                zip_ref.extractall(directory)
                zip_ref.close()


def main():
    """
    usage: casas_download.py [-h] [-d DIR] datasets [datasets ...]

    Download CASAS datasets to a directory.

    positional arguments:
      datasets           Dataset Names

    optional arguments:
      -h, --help         show this help message and exit
      -d DIR, --dir DIR  Directory to store datasets
    """
    parser = argparse.ArgumentParser(description='Download CASAS datasets to a directory.')
    parser.add_argument('-d', '--dir', help='Directory to store datasets')
    parser.add_argument('datasets', nargs='+', type=str, help='Dataset Names')
    args = parser.parse_args()
    dir_abs_path = os.path.abspath(os.path.expanduser(args.dir))
    if not os.path.isdir(dir_abs_path):
        user_input = input('Directory %s does not exist. Do you want to create it? [Y/n] ' %
                           dir_abs_path)
        if str.capitalize(user_input) == 'N':
            exit()
    CASAS_download(args.dir, tuple(args.datasets))

if __name__ == '__main__':
    main()

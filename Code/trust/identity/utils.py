# -*- coding: utf-8 -*-

"""
"""

import os
import random
import shutil


class ConnectionError(Exception):
    def __init__(self, value):
        r"""Connection error.
        """
        self.value = value

    def __str__(self):
        return repr(self.value)


def sshfs_connection():
    r"""Checks to see if you are connected to our data store through sshfs.

    Returns
    -------
    out : bool
        True if you are connected.
        False if you are not connected.

    """
    data = os.listdir('/vagrant/bsve_datastore')
    if not data:
        out = False
    if data:
        out = True
    return out


def get_random_file_paths(path, nmbr=1000):
    r"""Returns a random sampling of file paths based on the data in
    raw datastore on the server.

    Parameters
    ----------
    path : str
        The path to where the raw data is stored.
    nbr : {1000} int
        DEFAULT = 1000
        The number of files to return.

    Returns
    -------
    files : list
        A random sampling of file paths.

    """
    try:
        if sshfs_connection():
            file_paths = [os.path.abspath(os.path.join(p, F))
                          for p,d,f in os.walk(path)
                          for F in f if F.endswith('.json')]
            files = random.sample(file_paths, 1000)
            return files
    except:
        raise ConnectionError('Unable to establish a connection.')


def save_random_sample_localy(files):
    r"""Saves the list of given file paths locally to a data directory.

    """
    for f in files:
        basename = os.path.basename(f)
        dirname = '/'.join(['/vagrant'] + os.path.dirname(f).split('/')[-4:])
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        shutil.copy(f.replace('/Volumes', '/vagrant'),
                    os.path.join(dirname, basename))


def find_extension(path, ext):
    """Walks the directory structure of the given path and returns files with
    the given extension.

    Parameters
    ----------
    path : string
        The path to the directory to look for files in.
    ext : string
        The extension to look for.
    """
    _check_if_string(path)
    _check_if_string(ext)
    extension = ''
    if ext.startswith('.'):
        extension = ext
    else:
        extension = '.' + ext
    files = [os.path.abspath(os.path.join(p, F)) for p,d,f in os.walk(path)
             for F in f if F.endswith(extension)]
    return files



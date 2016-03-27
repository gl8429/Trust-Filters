# -*- coding: utf-8 -*-

"""
"""

import os
import json

import langdetect


def calculate(path):
    r"""Calculates the experience a user has with talking about flu.

    Parameters
    ----------
    path : string
        The path to the user generated content.

    Returns
    -------
    exp : float
        The experience of the user with generating flu content.

    Notes
    -----
    Experience is simply calculated as:

    .. math::

       experience = \frac{user generated content with flu words}
                         {all user generated content}

    """
    flu_posts = os.path.join(path, 'flu_posts_tokens.json')
    tokens = os.path.join(path, 'raw.json')
    if os.path.exists(flu_posts):
        with open(tokens, 'rb') as f:
            token_data = json.load(f)
        with open(flu_posts, 'rb') as f:
            flu_data = json.load(f)
        len_posts = len(token_data)
        len_flu = len(flu_data)
    exp = float(len_flu)/len_posts
    return exp

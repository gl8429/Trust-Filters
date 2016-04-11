Notes about NLP final projects
===============================

:Author: Guangyu Lin \& Huihuang Zheng
:Date: 3-26-2016

Thing to do
-----------
- Talk about projects (done)
- Setup the work repository (done)
- Upload related papers (done)
- Write some basic ideas (done)
- Upload previous code (done)
- Finish the draft of proposal (in progress)

- Familiar with Stanford NLP toolkit (in progress)
- Read papers and get to know some related algorithms (in progress)
- Define the data attributes that we need
- Design the basic algorithms

Trust Filters
-------------
Authority
^^^^^^^^^^^
- Talk about the disease
- Do not talk about their disease
- Measure is not a popularity measure

Experience
^^^^^^^^^^
- For a user, numeber of posts (twitter)

Expertise
^^^^^^^^^
- The probability of the content of topic for a specific user. 
       P(c_t|u) = P(u|c_t) * P(c_t)

Reputation
^^^^^^^^^^
- A popularity contest (PageRank)

Identity
^^^^^^^^
- Classify if a user is talking about themself or not    
    *Identify subject of a sentence*
     * (Me) I have the flu.
     * (Not Me) My kiddo is sick.
     * (Not Me) Ringo Star has the flu so no show.

    *Named Entity Recognizer*
     - Classifer, supervised
     - Standford NLP (We have tokenizer, NER, sentence recognizer)
     - Same synonyms example: kid, kiddo, child, ankle bitter
     - A no geotagged, B has geotagged, but B mention A in his tweet

    *Distance* 
     - Real measure of distance find data that are geotagged

Separation
^^^^^^^^^^
- Seperate different Named Entity Recognizer and absorb the relationship between them.
    *Range*
     - For different NER, their connections could range from 0 - 1, which is from `other` to `nuclear family`

  Example of Training NER

  +-------+---------------+
  |NER    |Relationship   |
  +=======+===============+
  |kid    |Nuclear Family |
  +-------+---------------+
  |child  |Nuclear Family |
  +-------+---------------+
  |grandpa|Extended Family|
  +-------+---------------+
  |me     |Me             |
  +-------+---------------+
  |you    |Not Me         |
  +-------+---------------+

    - This is going to be time consuming, due to the fact that we are going
      to have to label words to train a model. I've searched for training sets
      and they are surprisingly difficult to come by. We may have to just
      start training our own models with data we find is relevant.

Problem Solution Process
---------------------
*Two Problems*
 * Identify the subject(the person) of a sentence
 * Identify the relationship between the subject and the author
 
.. [#first] Collect raw text (exist dataset, exist data collectors, format attributes)
.. [#second] Design classifier like SVM, LDA Algorithm to filt related information (get rid of noise)
.. [#third] Raw text of the document is split into sentences (Sentence segmenter)
.. [#fourth] Each sentence is further subdivided into words, while stemming lemmatization remove punctuation normalizecase (data filters, tokenizer)
.. [#fifth] Each sentence is tagged with part-of-speech tags
.. [#sixth] Search for mentions of potentially interesting entities in each sentence (Name entity detection)
.. [#seventh] Search for likely relations between different entities in the test (relation detection)
.. [#eighth] Subject words predict (experiment and result)
.. image:: Figures/ie-architecture.png
            :width: 30

Useful Link of README's Format(rst)
-----------------------------------
Our README's format is reStructuredText(.rst), there are two useful handbook,`Sphinx Memo <http://rest-sphinx-memo.readthedocs.org/en/latest/ReST.html>`_
 `Sphinx CheatSheet <http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html>`_ ,for us to generate README below.

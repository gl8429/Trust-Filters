# -*- coding: utf-8 -*-

r""" trust.wordnet_graph_search utility developed by The University of Texas at
Austin for the Surety BioEvent App.

Description
-----------
Tree visualization for synset word relationships.
Not essential, but useful when exploring wordnet.

"""

# TODO: Check what happens when given a xnym that is definition or example?
# TODO: Track what relationship are, and label them (edges) in the tree.
# TODO: Refactor code, turn this into a class

from nltk.corpus import wordnet as wn


def search_tree(
        s, xnym=['hyponyms', 'meronyms'], infotype=['lemmas', 'definition'],
        maxdepth=10, condfun=lambda x: True,
        displaytree=True, h_spaces=2, v_spaces=1,
        emptyspace=False, show_one=True,
        cutoff_hypernyms=50, cutoff_hyponyms=8, def_cutoff=100):
    r""" Prints the tree structure of synsets (words) related to a seed
    synset. The relations can be hyponyms, hypernyms, meronyms, holonyms,
    particular kinds of meronyms and holonyms (part, substance and member),
    and combinations (union) of the above categories.

    Optionally also prints information next to each word in the tree: including
    definition, lemmas, meronyms, holonyms, examples, and combinations
    of the above.

    Usage
    -----
    search_tree(s, xnym = 'hyponyms', infotype = ['lemmas','meronyms'],\
                emptyspace=False)

    Parameters
    ----------
    s : str, or list of str, or list of Synsets
        The seed word(s) to build tree(s) from
    xnym : str, or list of str
        Specifies which kind of word-relations to search for from seed node.
        If str, can take values:
        'hyponyms', 'hypernyms', 'meronyms', 'holonyms',
        'part', 'part_meronyms', 'part_holonyms',
        'member','member_meronyms','member_holonyms',
        'substance','substance_meronyms', 'substance_holonyms'
        If list, the list contain any combination of the above
    infotype : str, or list of str
        Specifies which kind of information to display next to each synset
        If str, can take values:
        'name','definition','lemmas','lemma','examples',
        'hyponyms', 'hypernyms', 'meronyms','holonyms'
    condfun : function
        Function that takes synset as input and returns boolean. Can be used
        to only travel to synsets that satisfy another condition.
    displaytree : boolean
        If True it prints the tree
    emptyspace : boolean (optional)
        Whether to print tree with empty space to left or with full branches
        displayed ("---")
    show_one : boolean
        If show_one is False, won't display Hyper/hypo nym if there is only one
    v_spaces : int (optional)
        Amount of vertical space to print between nodes
    h_spaces : int (optional)
        Amount of horizontal space to print between nodes
    maxdepth : int
        Maximum distance to travel in wordnet graph
    cutoff_hypernyms : int
        Maximum number of hypernyms of synset to print
    cutoff_hyponyms : int
        Maximum number of hyponyms of synset to print
    def_cutoff : int
        Maximum number of characters in definition string to print

    Notes
    -----
    If s is a list, will print the trees structures for each Synset contained.

    Returns
    -------
    path(s) : list, or list of lists
        All synsets connected to seed node via specified relations (iterated)
        If several synset seeds were given, it just gives a flat list

    """

    def dfs(s, infs=[], path=[], condfun=condfun, indent=0):
        attribute = tuple(map(lambda x: get_attribute(s, x), infotype))
        # NOTE: need to use tuple instead of list  because it is immutable;
        # deepcopy took too long (when formatting attributes for printing)
        syns_to_follow = map(lambda x: get_attribute(s, x), xnym)
        syns_to_follow = [i for y in syns_to_follow for i in y]

        path = path + [s]
        infs = infs + [attribute]

        if displaytree:
            _print_line(s, attribute, infotype, show_one,
                        h_spaces, v_spaces, emptyspace, indent,
                        cutoff_hypernyms, cutoff_hyponyms, def_cutoff)

        for syn_item in syns_to_follow:

            condition = syn_item not in path and indent < maxdepth
            condition = condition and condfun(syn_item)

            if condition:
                path, infs = dfs(syn_item, infs, path, condfun,
                                 indent=indent+1)
        return path, infs

    allattrs = []
    allsyns = []

    if type(infotype) == str:
        infotype = [infotype]
    if type(xnym) == str:
        xnym = [xnym]
    if type(s) != list:
        s = [s]

    for syn in s:
        if type(syn) == str:
            syn = wn.synset(syn)
        if displaytree and len(s) > 1:
            print(syn.name()+" : "+syn.definition())

        syns, attrs = dfs(syn)

        if displaytree:
            print('_'*80)

        allsyns = allsyns + syns  # or flatten?
        allattrs = allattrs + attrs

    return allsyns, allattrs


def get_attribute(s, infotype):
    r"""
    Parameters
    ----------
    s : synset
        wordnet synset object whose attribute we want
    infotype: str
        specifies an attribute of s (hyponym, meronym, etc)

    Returns
    -------
    attribute : str
        attribute to print to screen
    """
    # Defined for ease of use
    if infotype in set(['meronyms', 'holonyms', 'part', 'substance', 'member',
                        'name']):
        # Control what is printed next to each synset
        if infotype == 'meronyms':
            attribute = s.substance_meronyms() + s.part_meronyms() + \
                        s.member_meronyms()
        elif infotype == 'holonyms':
            attribute = s.substance_holonyms() + s.part_holonyms() + \
                        s.member_holonyms()
        elif infotype == 'part':
            attribute = s.part_meronyms() + s.part_holonyms()
        elif infotype == 'substance':
            attribute = s.substance_meronyms() + s.substance_holonyms()
        elif infotype == 'member':
            attribute = s.member_meronyms() + s.member_holonyms()
        elif infotype == 'name':
            attribute = ''  # TODO: double-check this
        else:
            raise ValueError('wrong string for attribute')
    else:
        attribute = s.__getattribute__(infotype)()

    return attribute


def _print_line(
        s, attribute, infotype, show_one, h_spaces, v_spaces, emptyspace,
        indent, cutoff_hypernyms, cutoff_hyponyms, def_cutoff):
    r""" Prints one line, containing the synset and the attribute

    s : synset
        wordnet synset object whose attribute we want
    attribute : list of str
        attributes to print to screen
    infotype: str
        specifies an attribute of s (hyponym, meronym, etc)
    cutoff_hypernyms : int
        Maximum number of hypernyms of synset to print
    cutoff_hyponyms : int
        Maximum number of hyponyms of synset to print
    def_cutoff : int
        Maximum number of characters in definition string to print
    show_one : boolean
        If show_one is False, won't display Hyper/hypo nym if there is only one
    v_spaces : int (optional)
        Amount of vertical space to print between nodes
    h_spaces : int (optional)
        Amount of horizontal space to print between nodes
    emptyspace : boolean (optional)
        Whether to print tree with empty space to left or with full branches
        displayed ("---")
    indent : int
        Number of spaces to indent (how many edges down the tree from seed)
   """
    toprint = [None]*len(attribute)

    padding = ' |'+'-'*h_spaces
    spaces = h_spaces*{1: '  '+' ', 0: ' |'+' '}[emptyspace or v_spaces > 0]

    for i in range(len(infotype)):
        toprint[i] = _format_attribute(attribute[i], infotype[i], show_one,
                                       def_cutoff, cutoff_hyponyms,
                                       cutoff_hypernyms)

        if toprint[i] != "":
            toprint[i] = infotype[i] + ": "+toprint[i] + ". "

    print(((spaces * (indent+1)) + '\n')*v_spaces +
          spaces * (indent) + padding + s.name() + ": " + ''.join(toprint))


def _format_attribute(attribute, infotype, show_one,
                      def_cutoff, cutoff_hyponyms, cutoff_hypernyms):
    r"""
    Parameters
    ----------
    attribute : synset
        wordnet synset object whose attribute we want
    infotype: str
        specifies an attribute of s (hyponym, meronym, etc)
    show_one : boolean
        If show_one is False, won't display Hyper/hypo nym if there is only one
    cutoff_hypernyms : int
        Maximum number of hypernyms of synset to print
    cutoff_hyponyms : int
        Maximum number of hyponyms of synset to print
    def_cutoff : int
        Maximum number of characters in definition string to print

    Returns
    -------
    attribute : str
        attribute to print to screen, formatted

    """

    toprint = attribute

    if type(toprint) == list:
        if all(not isinstance(x, unicode) for x in toprint):
            toprint = [i.name() for i in toprint]
            if infotype == 'hypernyms':
                toprint = toprint[0:cutoff_hypernyms]
            if infotype == 'hyponyms':
                toprint = toprint[0:cutoff_hyponyms]

            if len(toprint) == 0:
                toprint = ''
            elif len(toprint) < len(attribute):
                toprint = (', '.join(toprint)) + '..'
            elif not show_one and len(toprint) == 1:
                toprint = ''
            else:
                toprint = ', '.join(toprint)
        else:
            toprint = ', '.join(toprint)

    if infotype == 'definition':
        toprint = toprint[0:def_cutoff]
        if len(toprint) < len(attribute):
            toprint = toprint + '..'

    return toprint

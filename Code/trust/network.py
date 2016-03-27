# -*- coding: utf-8 -*-

r"""
"""

import os

import networkx as nx

basepath = '/bsve_datastore/blogs/wordpress/done'


def load_graph(path):
    r"""
    """



def connected_graph():
    users = set(data.keys())
    for k,v in data.items():
        ltemp = set(v['likes'])
        ctemp = set(v['comments'])
        likes = users.intersection(ltemp)
        comments = users.intersection(ctemp)
        temp = {'connected_graph':
                {'likes': list(likes), 'comments': list(comments)}}
        data[k].update(temp)

# Read in author info.
basepath = 'bsve_datastore/blogs/wordpress/done'
blogs = [os.path.join(basepath, f) for f in os.listdir(basepath)
         if os.path.isdir(os.path.join(basepath, f))]
blogposts = []
length = len(blogs)
for i, blog in enumerate(blogs):
    postspath = os.path.join(blog, 'posts')
    postslist = [os.path.join(postspath, f)
                 for f in os.listdir(postspath) if f.endswith('.json')][0]
    blogposts.append(postslist)
    print '{0:0.2f}'.format(float((i+1))/length*100)





blogposts = [os.path.join(basepath, 'posts', f) for f in
             [ff for F in os.listdir(os.path.join(blogs, 'posts'))
              for ff in F if ff.endswith('.json')][0]]
authorpath = [os.path.join(f, 'author.json') for f in blogs
              if os.path.exists(os.path.join(f, 'author.json'))]
authors = {}
for author in authorpath:
    with open(author, 'r') as f:
        data = json.load(f)
    if len(data['wordpress']) != 1:
        print json.dumps(data, indent=4)
        break
        print len(data['wordpress'])
    authors[data['wordpress'][0]['site_ID']] = data['wordpress'][0]['nice_name']



G = nx.MultiDiGraph()
for k,v in data.items():
    likes = v['likes']
    comments = v['comments']
    connected_likes = v['connected_graph']['likes']
    connected_comments = v['connected_graph']['comments']
    if likes:
        for like in likes:
            G.add_edge(k, like, edge_type='likes')
    if comments:
        for comment in comments:
            G.add_edge(k, comment, edge_type='comments')
    if connected_likes:
        for like in connected_likes:
            G.add_edge(k, like, edge_type='connected_likes')
    if connected_comments:
        for comment in connected_comments:
            G.add_edge(k, comment, edge_type='connected_comments')


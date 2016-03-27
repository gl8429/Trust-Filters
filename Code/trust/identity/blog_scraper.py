from bs4 import BeautifulSoup as bs
import json

def scrubber(content_list):
    """Goes through the list of contents and removes html from the
    contents/translates unicode.  Then each content is broken up into
    paragraphs and saved to a list of paragraphs

    Returns a list of paragraphs"""

    paragraph_list = []
    for content in content_list:
        soup = bs(content)
        text = soup.find_all('p')
        text = map(lambda x: x.text.strip('\n').encode('ascii','ignore'), text)
        paragraph_list += text

    return paragraph_list

def get_paragraphs(path):
    """Opens the json file given by path, saves the contents to a list ,
    gets the paragraphs, scrubs the html, and returns a list of paragraphs

    Returns a list of paragraphs"""

    with open(path, 'rb') as post:
        data = json.loads(post.read())
        contents = [comment['content'] for comment in data['comments']]
        contents = contents[::-1]
        contents.append(data['content'])
        paragraphs = scrubber(contents)

    return paragraphs

def test():
    """This test simply scrubs the paragraphs from a blog given by a hard-
    coded path and returns the paragraphs"""

    filename = '/vagrant/data/blogs/wordpress/thebubblyspeckle.com/107.json'
    paragraphs = load_data(filename)
    return paragraphs

#paragraphs = test()
#print paragraphs

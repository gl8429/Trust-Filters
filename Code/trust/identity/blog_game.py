#
import sys
sys.path.insert(0, '/vagrant/bsve_datastore/code/dtra')
# sys.path.insert(0, '/vagrant/dtra')

#from utils import files
import utils
import os
import random
import blog_scraper
import json

if __name__ == '__main__':
    # give user instructions
    print "\nWelcome to the blog reading game! If this is your first time " +\
          "playing, begin by running the 'get_user_folders()' function. If " +\
          "you already have a user id and folder, begin by running the " +\
          "'user_interface()' function."


# only need to run this once!
def get_user_folders(users=2):
    """returns a folder of random blog post paths, one for each user"""

    # collect all json files below /vagrant/data
    p = '/'.join(os.path.abspath(utils.__file__).split('/')[:-4])
    path = os.path.join(p, 'data/blogs')
    f = utils.find_extension(path, 'json')

    # divide into random lists, one for each user
    random.shuffle(f)
    folders = [[] for i in range(users)]

    for idx, post_path in enumerate(f):
        folders[idx % users].append(post_path)

    # write lists to files
    for i in range(users):
        write_list_to_file(folders[i], get_filename(i))


def write_list_to_file(list, filename, codes='wb'):
    """writes a list to a file named filename"""

    f = open(filename, codes)
    f.writelines("%s\n" % post_path for post_path in list)
    f.close()


def get_filename(user_id):
    return 'user_lists/user_list' + str(user_id) + '.txt'


def get_post(user_list):
    """returns the path of a random post and removes it from the user's list"""

    if not user_list:
        print "Yay! There are no more blogs assigned to you."
        return ""
    else:
        post_path = user_list[0]
        user_list.pop(0)

        return post_path


def get_url(path):
    """gets url from json file"""

    with open(path, 'rb') as post:
        data = json.loads(post.read())
        url = data['URL']
    return url


def user_interface():
    """displays content one by one, has user decide category, saves content to
    appropriate category"""

    # get appropriate user list
    os.system('clear')
    user_id = raw_input("Hello! Please enter user id: ")
    with open(get_filename(user_id)) as f:
        user_list = f.read().splitlines()
    print "\033[1;41m" + "\nWARNING: Do not quit program in the middle of a" +\
          " blog post! Your work will not be saved. Please finish current " +\
          "post, then you may quit when prompted. If you would like to " +\
          "scrap the entire post, enter 'scrap' at any prompt." + "\033[0m"

    # classify a blog post.  continue pulling posts as long as the user wants.
    read_more = True
    while read_more:

        post_path = get_post(user_list)
        if not post_path:
            break
        paragraph_list = blog_scraper.get_paragraphs(post_path)
        me = []
        not_me = []
        scrap = False
        print "\033[1;34m" + "\nBlog path: " + post_path + "\033[0m"
        print "\033[1;34m" + "Blog url: " + get_url(post_path) + "\033[0m"

        for idx, paragraph in enumerate(paragraph_list):

            # ignore empty paragraphs
            if paragraph != '':

                # print paragraph
                l = len(paragraph_list)
                i = idx + 1
                print "\nHere is paragraph #%d out of %d.  Enjoy!\n" % (i, l)
                print "\033[0;32m" + paragraph + "\033[0m"
                print ""

                # get user classification
                prompt_user = True
                choice = raw_input("Which category? Enter 'me' for ME, 'not'" +
                                   " for NOT ME, or 'ignore' if neither " +
                                   "category fits\n")

                while prompt_user:
                    if choice == 'me':
                        me.append(paragraph)
                        prompt_user = False
                    elif choice == 'not':
                        not_me.append(paragraph)
                        prompt_user = False
                    elif choice == 'ignore':
                        prompt_user = False
                    elif choice == 'scrap':
                        scrap = True
                        break
                    else:
                        choice = raw_input("\033[1;41m" + "That's not an " +
                                           "option!! Please pay attention. " +
                                           "Enter 'me' for ME, 'not' for NOT" +
                                           " ME, or 'ignore' if neither " +
                                           "category fits." + "\033[0m\n")
                if scrap:
                    read_more = raw_input("\nPost scrapped. Would you like " +
                                          "to read through another post? " +
                                          "(y/n) ")
                    break

        # would the user like to keep playing?
        write_list_to_file(user_list, get_filename(user_id))
        if not scrap:
            write_list_to_file(me, 'ME.txt', codes='ab')
            write_list_to_file(not_me, 'NOT_ME.txt', codes='ab')
            read_more = raw_input("\nResponses recorded. Would you like to " +
                                  "read through another post? (y/n) ")
        if read_more == 'n':
            read_more = False
        else:
            read_more = True

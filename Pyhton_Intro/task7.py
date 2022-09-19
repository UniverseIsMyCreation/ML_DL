import re
def find_shortest(l):
    return min(map(len,re.sub('[^A-Za-z]',' ',l).split()),default=0)

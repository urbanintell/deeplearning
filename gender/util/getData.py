import requests
import urllib
import os
from bs4 import BeautifulSoup
from yaml import load, dump,load_all
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def getData(gender, startFrom = 0,startCount=0,showCount=True, showLink=True):
    searchingForImages = True
    j = startFrom
    i = startCount
    while searchingForImages:
        link = "http://www.thesartorialist.com/category/"+gender+"/page/" + str(j)
        r = requests.get(link)
        data = r.text
        soup = BeautifulSoup(data, "html5lib")
        
        if showLink:
            print("Link: " + link )

        number_pages, images = len(soup.find_all(
            class_="overhand")), soup.find_all(class_="overhand")
        if number_pages == 0:
            searchingForImages = False

        for k, v in enumerate(images):
            linkToImage = v.get('href')

            imageLink = requests.get(linkToImage)
            data = imageLink.text
            imageWebPage = BeautifulSoup(data, "html5lib")
            article = imageWebPage.find(class_="article-content")
            if article.a == None:
                continue
            imagepath = article.a.get('href')
            urllib.urlretrieve(imagepath, os.path.basename(gender+".{}.jpg".format(i)))
            if showCount:
                print (i)
            i+=1

        j += 1
    return j


#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   May 15 22:17:05 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Web Scrape
# @File:     Crawler 2.py
# @Software: PyCharm
"""
# Import Packages
import requests
from bs4 import BeautifulSoup

# Dealing with different website layout
class Content:
    def __init__(self,url,title,body):
        self.url = url
        self.title = title
        self.body = body

def getPage(url):
    """
    Utilty function used to get a Beautiful Soup object from a given URL
    """
    session = requests.Session()
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}
    try:
        req = session.get(url, headers=headers)
    except requests.exceptions.RequestException:
        return None
    bs = BeautifulSoup(req.text, 'html.parser')
    return bs

# Cannot extract the data from website:
def scrapeNYTime(url):
    bs = getPage(url)
    title = bs.find('h1')
    lines = bs.select('div.StoryBodyCompanionColumn div p')
    body = '\n'.join([line.text for line in lines])
    return Content(url,title,body)

def scapeBrookings(url):
    bs = getPage(url)
    title = bs.find('h1').text
    body = bs.find('div', {'class':'post-body'}).text
    return Content(url,title,body)

# Brookings
url1 = 'https://www.brookings.edu/blog/future-development' \
      '/2018/01/26/delivering-inclusive-urban-access-3-uncomfortable-truths/'

content1 = scapeBrookings(url1)
print('Title: {}'.format(content1.title))
print('URL: {}\n'.format(content1.url))
print(content1.body)

# NYTimes
url2 = 'https://www.nytimes.com/2018/01/25/opinion/sunday/silicon-valley-immortality.html' # website has response

try:
    content2 = scrapeNYTime(url2)
except AttributeError as e:
    print('Function error:',e)


content2 = scrapeNYTime(url2)
print('Title: {}'.format(content2.title)) # Server issue "Forbidden 403"
print('URL: {}\n'.format(content2.url)) # Correct URL
print(content2.body) # Nothing content

# Detail of title
# <h1 id="link-2b8abacb" class="css-1qxijs e1h9rw200" data-testid="headline">The Men Who Want to Live Forever</h1>


class Content:
    '''
    Common base class for all articles/pages
    '''

    def __init__(self, url, title, body):
        self.url = url
        self.title = title
        self.body = body

    def print(self):
        '''
        Flexible printing function controls output
        '''
        print('URL: {}'.format(self.url))
        print('TITLE: {}'.format(self.title))
        print('BODY:\n{}'.format(self.body))

class Website:
    '''
    Contains information about the website structure
    '''
    def __init__(self,name,url,titleTag,bodyTag):
        self.name = name
        self.url = url
        self.titleTag = titleTag
        self.bodyTag = bodyTag

class Crawler:

    def getPage(self, url):
        try:
            req = requests.get(url)
        except requests.exceptions.RequestException:
            return None
        return BeautifulSoup(req.text, 'html.parser')

    def safeGet(self, pageObj, selector):
        """
        Utilty function used to get a content string from a Beautiful Soup
        object and a selector. Returns an empty string if no object
        is found for the given selector
        """
        selectedElems = pageObj.select(selector)
        if selectedElems is not None and len(selectedElems) > 0:
            return '\n'.join([elem.get_text() for elem in selectedElems])
        return ''

    def parse(self, site, url):
        """
        Extract content from a given page URL
        """
        bs = self.getPage(url)
        if bs is not None:
            title = self.safeGet(bs, site.titleTag)
            body = self.safeGet(bs, site.bodyTag)
            if title != '' and body != '':
                content = Content(url, title, body)
                content.print()

# Setting model
crawler = Crawler()

siteData = [
    ['O\'Reilly Media', 'http://oreilly.com', 'h1', 'section#product-description'],
    ['Reuters', 'http://reuters.com', 'h1', 'div.StandardArticleBody_body_1gnLA'],
    ['Brookings', 'http://www.brookings.edu', 'h1', 'div.post-body'],
    ['New York Times', 'http://nytimes.com', 'h1', 'div.StoryBodyCompanionColumn div p']
]
websites = []
for row in siteData:
    websites.append(Website(row[0], row[1], row[2], row[3]))

crawler.parse(websites[0], 'http://shop.oreilly.com/product/0636920028154.do')
crawler.parse(websites[1], 'http://www.reuters.com/article/us-usa-epa-pruitt-idUSKBN19W2D0')
crawler.parse(websites[2], 'https://www.brookings.edu/blog/techtank/2016/03/01/idea-to-retire-old-methods-of-policy-education/')
crawler.parse(websites[3],'https://www.nytimes.com/2018/01/28/business/energy-environment/oil-boom.html')
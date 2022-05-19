#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   May 15 12:35:32 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Web Scrape
# @File:     Crawler 1.py
# @Software: PyCharm
"""
# Import Packages
from urllib.request import urlopen
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import datetime
import random
import re

kb = urlopen('https://en.wikipedia.org/wiki/Kevin_Bacon')
kb_bs = BeautifulSoup(kb, 'html.parser')
'''
for link in kb_bs.find_all('a'):
    if 'href' in link.attrs:
        print(link.attrs['href'])

'''

# used regex to select the information
for link in kb_bs.find('div', {'id':'bodyContent'}).find_all('a', href=re.compile('^(/wiki/)((?!:).)*$')):
    if 'href' in link.attrs:
        print(link.attrs['href'])

# getLink function
'''
 1. getLink, a function takes in a Wikipedia article URL of the from/wiki/<Article_Name> and returns a list of all
    linked article URLs in the same form;
 2. Choose a random article link from the returned list, and call getLinks again, utill stop the program or until 
    not article links are found on the new pages.
'''
random.seed(datetime.datetime.now())
def getLinks(articleURL):
    html = urlopen('https://en.wikipedia.org{}'.format(articleURL))
    bs = BeautifulSoup(html, 'html.parser')
    result = bs.find('div', {'id':'bodyContent'}).find_all(
        'a', href=re.compile('^(/wiki/)((?!:).)*$'))
    return result

links = getLinks('/wiki/Kevin_Bacon')
while len(links) > 0:
    newArticle = links[random.randint(0, len(links)-1)].attrs['href']
    print(newArticle)
    links = getLinks(newArticle)

# make sure did not crawl twice for each website
pages = set()
def getLinks(pageURL):
    global pages
    html = urlopen('https://en.wikipedia.org{}'.format(pageURL))
    bs = BeautifulSoup(html, 'html.parser')
    for link in bs.find_all('a', href=re.compile('^(/wiki/)')):
        if 'href' in link.attrs:
            if link.attrs['href'] not in pages:
                newPage = link.attrs['href']
                print(newPage)
                pages.add(newPage)
                getLinks(newPage)
getLinks('')

# Collecting data across and entire site
pages = set()
def getLinks(pageURL):
    global pages
    html = urlopen('https://en.wikipedia.org{}'.format(pageURL))
    bs = BeautifulSoup(html, 'html.parser')
    try:
        print(bs.h1.get_text())
        print(bs.find(id='mw-content-text').find_all('p')[0])
        print(bs.find(id='ca-edit').find('span').find('a').attrs['href'])
    except AttributeError:
        print('This page is missing something! Continuing.')

    for link in bs.find_all('a',href=re.compile('^(/wiki/)')):
        if 'href' in link.attrs:
            if link.attrs['href'] not in pages:
                newPage = link.attrs['href']
                print('-'*20)
                print(newPage)
                pages.add(newPage)
                getLinks(newPage)
getLinks('')

# Crawling across the Internet
pages = set()
random.seed(datetime.datetime.now())

# Internal Link found on a poge
def getInternalLinks(bs, includeURL):
    includeURL = '{}://{}'.format(urlparse(includeURL).scheme,
                                  urlparse(includeURL).netloc)
    internalLinks = []
    # Find all links begin with '/'
    for link in bs.find_all('a',href=re.compile('^(/|.*'+includeURL+')')):
        if link.attrs['href'] is not None:
            if link.attrs['href'] not in internalLinks:
                if (link.attrs['href'].startswith('/')):
                    internalLinks.append(includeURL+link.attrs['href'])
                else:
                    internalLinks.append(link.attrs['href'])
    return internalLinks

# External link found on a page
def getExternalLinks(bs, externalURL):
    externalLinks = []
    # Find add links start with 'http' that do not contain the current URl
    for link in bs.find_all('a', href=re.compile('^(http|www)((?!'+externalURL+').)*')):
        if link.attrs['href'] is not None:
            if link.attrs['href'] not in externalLinks:
                externalLinks.append(link.attrs['href'])
    return externalLinks

def getRandomExternalLink(startPage):
    html = urlopen(startPage)
    bs = BeautifulSoup(html, 'html.parser')
    externalLinks = getExternalLinks(bs, urlparse(startPage).netloc)
    if len(externalLinks) == 0:
        print('No external links, looking around the site for one')
        domain = '{}://{}'.format(urlparse(startPage).scheme, urlparse(startPage).netloc)
        internalLink = getInternalLinks(bs, domain)
        return getExternalLinks(internalLink[random.randint(0,len(internalLink)-1)])
    else:
        return externalLinks[random.randint(0,len(externalLinks)-1)]

def followExternalOnly(startSite):
    externalLink = getRandomExternalLink(startSite)
    print('Random external link is: {}'.format(externalLink))
    followExternalOnly(externalLink)

followExternalOnly('http://oreilly.com')

# Collect a list of all external URLs found on the site:
allExtLinks = set()
allIntLinks = set()

def getAllExternalLinks(siteURL):
    html = urlopen(siteURL)
    domain = '{}://{}'.format(urlparse(siteURL).scheme, urlparse(siteURL).netloc)
    bs = BeautifulSoup(html, 'html.parser')
    internalLinks = getInternalLinks(bs, domain)
    externalLinks = getExternalLinks(bs, domain)

    for link in externalLinks:
        if link not in allExtLinks:
            allExtLinks.add(link)
            print(link)
    for link in internalLinks:
        if link not in allIntLinks:
            allIntLinks.add(link)
            getAllExternalLinks(link)

allIntLinks.add('https://www.deeplearningbook.org')
getAllExternalLinks('https://www.deeplearningbook.org')
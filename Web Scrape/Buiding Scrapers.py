#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   May 14 12:12:06 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Web Scrape
# @File:     Buiding Scrapers.py
# @Software: PyCharm
"""
# Import Packages
from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.error import URLError
from bs4 import BeautifulSoup

# URL Address: https://www.pythonscraping.com/pages/

# basic read
html = urlopen('http://www.pythonscraping.com/pages/page1.html')
bs = BeautifulSoup(html, 'html.parser')
print(bs.h1)

# error test
try:
    html = urlopen('http://pythonscrapingthisdoesnotexist.com')
except HTTPError as e:
    print(e)
except URLError as e:
    print('This server could not be found!')
else:
    print('It worked!')


try:
    badContent = bs.nonExistingTag.anotherTag
except AttributeError as e:
    print('Tag was not found')
else:
    if badContent == None:
        print('Tag was not found')
    else:
        print(badContent)

# Advance BeautifulSoup
warandpeace = urlopen('https://www.pythonscraping.com/pages/warandpeace.html')
bs = BeautifulSoup(warandpeace.read(), 'html.parser')
nameList = bs.find_all('span', {'class':'green'})
for name in nameList:
    print(name.get_text())

# Navigating Tree
gift = urlopen('https://www.pythonscraping.com/pages/page3.html')
bs = BeautifulSoup(gift, 'html.parser')

for child in bs.find('table', {'id':'giftList'}).children: # using find() instead of find_all()
    print(child)

## sibling
for sibling in bs.find('table', {'id':'giftList'}).tr.next_siblings:
    print(sibling)

## parent
print(bs.find('img',{'src':'../img/gifts/img1.jpg'}).parent.previous_sibling.get_text())

# regex
### Email Identification
import re
email = re.match(r'[A-Za-z0-9\._+]+@[A-Za-z]+\.(com|edu|org|net|gov)')

### Version 1
def email_test1(email):
    if re.match(r'[A-Za-z0-9\._+]+@[A-Za-z]+\.(\w+)', email):
        print('This is a correct email format!')
    else:
        print('This is NOT a correct email format!')
### Version 2
def email_test2(email):
    if re.match(r'[\w.]+@[A-Za-z0-9]+\.(\w+)', email):
        print('This is a correct email format!')
    else:
        print('This is NOT a correct email format!')

email_test2('zening.ye@gmail.com')
email_test2('zening.ye@q_1.com')

images = bs.find_all('img', {'src':re.compile('\.\.\/img\/gifts/img.*\.jpg')})
for image in images:
    print(image['src'])

# Lambda Expressions
bs.find_all(lambda tag: tag.get_text() == 'Or maybe he\'s only resting?')
bs.find_all('', text='Or maybe he\'s only resting?')
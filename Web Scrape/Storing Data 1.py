#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   May 19 22:53:31 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Summer2022
# @File:     Storing Data 1.py
# @Software: PyCharm
"""
# Import Packages
import os
from urllib.request import urlretrieve,urlopen
import requests
from bs4 import BeautifulSoup

# single download image from the webpage
path = urlopen('https://www.tensorflow.org') # equivalent to path = requests.get('https://www.tensorflow.org')
bs = BeautifulSoup(path, 'html.parser') # bs = BeautifulSoup(path.text, 'html.parser')
imageLocation = bs.find('img')['src']
# save the image into logo.jpg under Summer2022
urlretrieve (imageLocation, 'logo.svg')


# multiple download images (Tensorflow)
def getURL(baseUrl, source):
    if source.startswith('https://www.'):
        url = source
    else:
        url = f'{baseUrl}{source}'
    return url

def saveDir(baseUrl, absoluteUrl, downloadDirectory):

    # for url contain www.gstatic.com replace with another format
    if 'www.gstatic.com' in absoluteUrl:
        path = absoluteUrl.replace('https://www.gstatic.com/devrel-devsite/prod/vda9a852fe58dc4f0a77df9bfbfef645e053a541851391590524ef926ac0c5e1c','')
    else:
        path = absoluteUrl.replace((baseUrl+'/site-assets/images/marketing'), '')
    path = downloadDirectory + path
    directory = os.path.dirname(path)

    # create directary for download files
    if not os.path.exists(directory):
        os.makedirs(directory)

    return path

# setting download direction
downloadDirectory = 'Web Scrape/tensoflow-img'
baseUrl = 'https://www.tensorflow.org'

# request html
html = requests.get('https://www.tensorflow.org')
bs = BeautifulSoup(html.text, 'html.parser')
downloadList = bs.find_all(src=True)
del downloadList[1], downloadList[1]

# create html list
ls = []
for download in downloadList:
    fileUrl = getURL(baseUrl, download['src'])
    ls.append(fileUrl)
    # if fileUrl is not None:
    #     print(fileUrl)

# save the images
for i in range(0,len(ls)):
    urlretrieve(ls[i], saveDir(baseUrl, ls[i], downloadDirectory))


'''
class getURL:

    # store the raw data
    def __init__(self,baseUrl,source,absoluteUrl,downloadDirectory):
        self.baseUrl = baseUrl
        self.source = source
        self.absoluteUrl = absoluteUrl
        self.downloadDirectory = downloadDirectory

    def getAbsoluteURL(self, baseUrl, source):
    if source.startswith('https://www.'):
        url = source
    else:
        url = f'{baseUrl}{source}'
    return url
    
    def saveDir(self, baseUrl, absoluteUrl, downloadDirectory):
    if 'www.gstatic.com' in absoluteUrl:
        path = absoluteUrl.replace('https://www.gstatic.com/devrel-devsite/prod/vda9a852fe58dc4f0a77df9bfbfef645e053a541851391590524ef926ac0c5e1c','')
    else:
        path = absoluteUrl.replace((baseUrl+'/site-assets/images/marketing'), '')
    path = downloadDirectory + path
    directory = os.path.dirname(path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    return path
'''
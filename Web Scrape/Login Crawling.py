#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   May 27 11:35:42 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Web Scrape
# @File:     Login Crawling.py
# @Software: PyCharm
"""
# Import Packages
import requests

# Login with user name
params = {'firstname': 'Zening', 'lastname': 'Ye'}
r = requests.post('https://pythonscraping.com/pages/files/processing.php',data=params)
print(r.text)

# upload file (error with upload file, server problem not the code)
logo = open('tensoflow-img/logo.svg','rb')
file = {'uploadFile':logo}
r = requests.post('https://pythonscraping.com/pages/files/processing2.php',data=file)
print(r.text)

# Login via cookies
params = {'username': 'Shoko Takahashi', 'password': 'password'}
r = requests.post('https://pythonscraping.com/pages/cookies/welcome.php',params)
print('Cookie is set to:\n', r.cookies.get_dict())
print('Going to profile page......')
r = requests.get('https://pythonscraping.com/pages/cookies/profile.php',cookies=r.cookies)
print(r.text)

## Better dealing with cookies by using requests.Session()
params = {'username': 'Shoko Takahashi', 'password': 'password'}
session = requests.Session()
s = session.post('https://pythonscraping.com/pages/cookies/welcome.php',params)
print('Cookie is set to:\n', s.cookies.get_dict())
print('Going to profile page......')
s = session.get('https://pythonscraping.com/pages/cookies/profile.php')
print(s.text)

# HTTP Authentication
from requests.auth import AuthBase, HTTPBasicAuth

auth = HTTPBasicAuth('Shoko','password')
r = requests.post('https://pythonscraping.com/pages/auth/login.php',auth=auth)
print(r.text)


asus = HTTPBasicAuth('znye','Ye990131')
asus_s = requests.session().post('http://router.asus.com/Main_Login.asp',auth=asus)
print(asus_s.text)
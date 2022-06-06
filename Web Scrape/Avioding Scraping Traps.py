#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 03 22:11:46 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Web Scrape
# @File:     Avioding Scraping Traps.py
# @Software: PyCharm
"""
# Import Packages
import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

# Using header
session = requests.Session()
header = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5)'\
           'AppleWebKit 537.36 (KHTML, like Gecko) Chrome',
           'Accept':'text/html,application/xhtml+xml,application/xml;'\
           'q=0.9,image/webp,*/*;q=0.8'}
html = 'https://www.whatismybrowser.com/developers/what-http-headers-is-my-browser-sending'
req = session.get(html, headers=header)
bs = BeautifulSoup(req.text, 'html.parser')
print(bs.find('table',{'class':'table-striped'}).get_text())

# Handling cookies with JavaScript
driver_option = webdriver.ChromeOptions()
driver_option.add_argument('--incognito')
driver_option.add_argument('headless')
driver = webdriver.Chrome(options=driver_option)
driver.get('https://pythonscraping.com/')
driver.implicitly_wait(1)
cookies = driver.get_cookies()
time.sleep(1) # act like a human
print(cookies)
driver.quit()
driver.close()

driver2 = webdriver.Chrome(options=driver_option)
driver2.get('https://pythonscraping.com/')
driver2.delete_all_cookies()
for cookie in cookies:
    if not cookie['domain'].startswith('.'):
        cookie['domain'] = f".{cookie['domain']}"
    driver2.add_cookie(cookie)

driver2.get('https://pythonscraping.com')
driver2.implicitly_wait(1)
print(driver2.get_cookies())
driver2.quit()

# Avioding Honeypots
driver3 = webdriver.Chrome(options=driver_option)
driver3.get('https://pythonscraping.com/pages/itsatrap.html')
links = driver3.find_elements(By.TAG_NAME,'a')
for link in links:
    if not link.is_displayed():
        print(f"The link {link.get_attribute('href')} is a trap.")
fields = driver3.find_elements(By.TAG_NAME, 'input')
for field in fields:
    if not field.is_displayed():
        print(f"Do not change value of {field.get_attribute('name')}.")
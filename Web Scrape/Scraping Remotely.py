#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 08 16:36:46 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Web Scrape
# @File:     Scraping Remotely.py
# @Software: PyCharm
"""
# Import Packages
from selenium import webdriver
from selenium.webdriver import ChromeOptions

driver_options = ChromeOptions()
driver_options.add_argument('--incognito')
driver_options.add_argument('headless')
driver_options.add_argument('"--proxy-server=socks5://127.0.0.1:9150"')
driver = webdriver.Chrome(options=driver_options)

driver.get('https://icanhazip.com/')
print(driver.page_source)
driver.close()

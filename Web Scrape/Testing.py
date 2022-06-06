#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 06 13:28:22 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Web Scrape
# @File:     Testing.py
# @Software: PyCharm
"""
# Import Packages
import unittest
import requests
import re
import random
from bs4 import BeautifulSoup
from urllib.parse import unquote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys


###############################################
class TestAddition(unittest.TestCase):

    def setUp(self):
        print('Setting up the test')

    def tearDown(self):
        print('Tearing down the test')

    def test_twoPlustwo(self):
        total = 2 + 2
        self.assertEqual(4, total)
#
# if __name__ == '__main__':
#     unittest.main()
###############################################

###############################################
# Testing Wikipedia

class TestWikipedia(unittest.TestCase):
    bs = None
    def setUpClass():
        url = 'https://en.wikipedia.org/wiki/Monty_Python'
        TestWikipedia.bs = BeautifulSoup(requests.get(url).text, 'html.parser')

    def test_titleText(self):
        pageTitle = TestWikipedia.bs.find('h1').get_text()
        self.assertEqual('Monty Python', pageTitle)

    def test_contentExists(self):
        content = TestWikipedia.bs.find('div', {'id':'mw-content-text'})
        self.assertIsNotNone(content)

# if __name__ == '__main__':
#     unittest.main()
###############################################

###############################################
# run the test multiple times

class TestWikipedia(unittest.TestCase):

    def test_PageProperty(self):
        self.url = 'https://en.wikipedia.org/wiki/Monty_Python'
        for i in range(1,10): # test the first 10 pages
            self.bs = BeautifulSoup(requests.get(self.url).text, 'html.parser')
            titles = self.titleMatchesURL()
            self.assertEqual(titles[0], titles[1])
            self.assertTrue(self.contentExists())
            self.url = self.getNextLink()
        print('Test Complete!')

    def titleMatchesURL(self):
        pageTitle = self.bs.find('h1').get_text()
        urlTitle = self.url[(self.url.index('/wiki/')+6):]
        urlTitle = urlTitle.replace('_', ' ')
        urlTitle = unquote(urlTitle)
        return [pageTitle.lower(), urlTitle.lower()]

    def contentExists(self):
        content = self.bs.find('div', {'id':'mw-content-text'})
        if content is not None:
            return True
        return False

    def getNextLink(self):
        # return random link on page
        links = self.bs.find('div',{'id':'bodyContent'}).find_all('a',
                                                                  href=re.compile('^(/wiki/)((?!:).)*$'))
        randomLink = random.SystemRandom().choice(links)
        return f"https://wikipedia.org{randomLink.attrs['href']}"

# if __name__ == '__main__':
#     unittest.main()
###############################################

###############################################
# Testing with Selenium
## setup driver
driver_options = webdriver.ChromeOptions()
driver_options.add_argument('--incognito')
driver_options.add_argument('headless')
driver = webdriver.Chrome(options=driver_options)

## load website
driver.get('https://en.wikipedia.org/wiki/Monty_Python')
assert 'Monty Python' in driver.title
driver.close()

driver.get('https://pythonscraping.com/pages/files/form.html')

firstnameField = driver.find_element(By.NAME,'firstname')
lastnameField = driver.find_element(By.NAME,'lastname')
submitButton = driver.find_element(By.ID, 'submit')

## Method 1
firstnameField.send_keys('Shoko')
lastnameField.send_keys('Takahashi')
submitButton.click()

## Method 2
actions = ActionChains(driver).\
    click(firstnameField).send_keys('Shoko').\
    click(lastnameField).send_keys('Takahashi').\
    send_keys(Keys.RETURN)
actions.perform()

## print result
print(driver.find_element(By.TAG_NAME, 'body').text)

driver.close()

# Interaction with Selenium (drag and drop)
driver.get('https://pythonscraping.com/pages/javascript/draggableDemo.html')
print(driver.find_element(By.ID,'message').text) # print the performance before action

element = driver.find_element(By.ID,'draggable')
target = driver.find_element(By.ID, 'div2')
actions = ActionChains(driver)
actions.drag_and_drop(element, target).perform()

print(driver.find_element(By.ID,'message').text) # print the performance after action

# Take screenshot
driver.get('https://tensorflow.org')
driver.get_screenshot_as_file('tf-screenshoot.png')
###############################################

###############################################
# Tesing for drag and drop
class TestDragandDrop(unittest.TestCase):
    driver = None
    driver_options = webdriver.ChromeOptions()
    driver_options.add_argument('--incognito')
    driver_options.add_argument('headless')
    def setUp(self):
        self.driver = webdriver.Chrome(options=self.driver_options)
        url = 'https://pythonscraping.com/pages/javascript/draggableDemo.html'
        self.driver.get(url)

    def tearDown(self):
        print('Tearing down the test')

    def test_drag(self):
        element = self.driver.find_element(By.ID, 'draggable')
        target = self.driver.find_element(By.ID, 'div2')
        actions = ActionChains(self.driver)
        actions.drag_and_drop(element, target).perform()
        self.assertEqual('You are definitely not a bot!', self.driver.find_element(By.ID, 'message').text)

# if __name__ == '__main__':
#     unittest
###############################################
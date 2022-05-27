#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   May 20 16:49:41 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Web Scrape
# @File:     Storing Data 2.py
# @Software: PyCharm
"""
# Import Packages
import csv
import re
import requests
from bs4 import BeautifulSoup

# store file into CSV
csvFile = open('Web Scrape/test.csv', 'w+')
try:
    writer = csv.writer(csvFile)
    writer.writerow(('number', 'number plus 2', 'number times 2'))
    for i in range(10):
        writer.writerow((i, i+2, i*2))
finally:
    csvFile.close()

# retrieve an HTML table and write into csv
html = requests.get('https://en.wikipedia.org/wiki/Comparison_of_text_editors')
bs = BeautifulSoup(html.text,'html.parser')
table = bs.find_all('table',{'class':re.compile(r'wikitable$')})[0]
rows = table.find_all('tr')


# write csv
csvFile = open('Web Scrape/editors.csv', 'wt+')
writer = csv.writer(csvFile)
try:
    for row in rows:
        csvRow = []
        for cell in row.find_all(re.compile(r'(td|th)')):
            csvRow.append(cell.get_text())
        writer.writerow(csvRow)
finally:
    csvFile.close()
# the format in csv need to change if you want to see the full table.

# connect with SQLite3 Database
import sqlite3
import random

# Create database
conn = sqlite3.connect('Web Scrape/Scraping.db')
cur = conn.cursor()

# Create table (strongly recommend use DataGrip for database file!!!!)
cur.execute("create table pages(id integer primary key AUTOINCREMENT, title varchar(200), "
            "content varchar(10000), created TIMESTAMP default (datetime('now','localtime')))")
cur.execute("insert into pages(id, title, content,created) values (0,'Test page title',"
            "'This is some test page content. It can be up to 10,000 characters long.',(datetime('now','localtime')))")
cur.execute('select * from pages where id == 0')
print(cur.fetchone())
conn.commit() # Always remember commit after any execution!!!!!!!
cur.close()
conn.close()
# cur.execute('select id, title from pages where content like "%page content%"')
# print(cur.fetchone())
# cur.execute('drop table pages')


# store data into database
conn = sqlite3.connect('Web Scrape/Scraping.db')
cur = conn.cursor()

# Delete data
cur.execute('delete from pages where id == 0')
conn.commit()

## Store function
def store(title, content):
    cur.execute("INSERT INTO pages(title,content) VALUES (?,?)", (str(title),str(content),))
    conn.commit()

## Get information from URL
def getLinks(articleUrl):
    html = requests.get('https://en.wikipedia.org' + articleUrl)
    bs = BeautifulSoup(html.text, 'html.parser')
    title = bs.find('h1').get_text()
    content = bs.find('div', {'id':'mw-content-text'}).get_text()
    store(title, content)
    return bs.find('div',{'id':'bodyContent'}).find_all('a',href=re.compile('^(/wiki/)((?!:).)*S'))

links = getLinks('/wiki/kevin_Bacon')
try:
    while len(links) > 0: # it will run a long time
         newArticle = links[random.randint(0, len(links)-1)].attrs['href']
         print(newArticle)
         links = getLinks(newArticle)
finally:
    cur.close()
    conn.close()
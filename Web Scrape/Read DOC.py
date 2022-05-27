#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   May 25 11:36:24 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Web Scrape
# @File:     Read DOC.py
# @Software: PyCharm
"""
# Import Packages
import requests
import PyPDF2
import io
import pandas as pd
from bs4 import BeautifulSoup

# Text
textPages = requests.get('https://www.pythonscraping.com/pages/warandpeace/chapter1.txt')
textPages.text

textPages = requests.get('https://www.pythonscraping.com/pages/warandpeace/chapter1-ru.txt')
textPages.text.encode('utf-8')

html = requests.get('https://en.wikipedia.org/wiki/Python_(programming_language)')
bs = BeautifulSoup(html.text, 'html.parser')
content = bs.find('div', {'id':'mw-content-text'}).get_text()
content = bytes(content, 'UTF-8')
content = content.decode('UTF-8')

# CSV (Normally use pandas is enough)
data = pd.read_csv('https://pythonscraping.com/files/MontyPythonAlbums.csv')
for row in range(len(data)):
    print(data.loc[row,:].to_dict()) # print data as a dictionary

# PDF
pdfFile = requests.get('https://pythonscraping.com/pages/warandpeace/chapter1.pdf')
pdfFile_bytes = io.BytesIO(pdfFile.content)
pdfOutput = PyPDF2.PdfReader(pdfFile_bytes)
text = ''
for page in range(pdfOutput.numPages):
    obj = pdfOutput.getPage(page)
    text = text + obj.extract_text()
    print(text)

## store PDF content into SQL database
import sqlite3
conn = sqlite3.connect('PDF_store.db')
cur = conn.cursor()

cur.execute('create table content(Chapter varchar(20), Content varchar(5000))')
cur.execute("insert into content(Chapter, Content) values ('Chapter 0',"
            "'This is test conent for database')")
conn.commit()

cur.execute("insert into content(Chapter, Content) values(?,?)",(text[0:9],text[9:],))
conn.commit()
cur.close()
conn.close()

# Office .docx
from zipfile import ZipFile
wordFile = requests.get('https://pythonscraping.com/files/AWordDocument.docx')
wordFile = io.BytesIO(wordFile.content)
document = ZipFile(wordFile)
content = document.read('word/document.xml')

wordobj = BeautifulSoup(content.decode('utf-8'),'xml')
textStrings = wordobj.find_all('w:t')

for textElem in textStrings:
    print(textElem.text)
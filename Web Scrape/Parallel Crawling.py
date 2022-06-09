#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 07 14:04:04 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Web Scrape
# @File:     Parallel Crawling.py
# @Software: PyCharm
"""
# Import Packages
import _thread
import time
import requests
from bs4 import BeautifulSoup
import re
import random
import sqlite3
from queue import Queue

# Using multiple threads
def print_time(threadName, delay, iterations):
    start = int(time.time())
    for i in range(0, iterations):
        time.sleep(delay)
        seconds_elapsed = str(int(time.time()) - start)
        print(f'{seconds_elapsed} {threadName}')

try:
    _thread.start_new_thread(print_time, ('Fizz', 3, 33))
    _thread.start_new_thread(print_time, ('Buzz', 5, 20))
    _thread.start_new_thread(print_time, ('Counter', 1, 100))
except:
    print('Error: unable to start thread')

while 1:
    pass

# Basic Crawling with multithreads
## visited = [] # add a list to ensure only seen once for each website
def get_links(thread_name, bs):
    print(f'Getting links in {thread_name}')
    return bs.find('div',{'id':'bodyContent'}).find_all('a',href=re.compile('^(/wiki/)((?!:).)*$'))
    '''
    links = bs.find('div',{'id':'bodyContent'}).find_all('a',href=re.compile('^(/wiki/)((?!:).)*$'))
    return [link for link in links if link not in visited]
    '''


## Define function for thread
def scrape_article(thread_name, path):
    # visited.append(path)
    html = requests.get(f'https://en.wikipedia.org{path}')
    time.sleep(5)
    bs = BeautifulSoup(html.text, 'html.parser')
    title = bs.find('h1').get_text()
    print(f'Scraping {title} in thread {thread_name}')
    links = get_links(thread_name, bs)
    if len(links) > 0:
        newArticle = links[random.randint(0, len(links) - 1)].attrs['href']
        print(newArticle)
        scrape_article(thread_name, newArticle)

## Create two thread for the program
try:
    _thread.start_new_thread(scrape_article, ('Thread 1', '/wiki/Kevin_Bacon',))
    _thread.start_new_thread(scrape_article, ('Thread 2', '/wiki/Monty_Python',))
except:
    print('Error: unable to start threads')
while 1:
    pass

# Storing into database with Queue
## Connect database and create new table
conn = sqlite3.connect('Scraping.db')
cur = conn.cursor()

'''
cur.execute("create table Queue(ID integer primary key AUTOINCREMENT, Title varchar(200), "
            "Path varchar(10000), Created TIMESTAMP default (datetime('now','localtime')))")
cur.execute("insert into Queue(Title,Path) values (?,?)",('Test','Undefine Path',)) # testing
conn.commit()
'''
cur.close()
conn.close()
'''
Never use cur.rowcount in SQLite3!!!! 
Use len(cur.fetchall()) or list(cur.fetchall())
'''

## Define storing function
def storage(queue):
    conn = sqlite3.connect('Scraping.db')
    cur = conn.cursor()
    print('Connected to the database, ready for further operation.')

    while 1:
        if not queue.empty():
            article = queue.get()
            article_path = (article['path'],)
            cur.execute('select * from Queue where Path = ?', article_path)
            if len(cur.fetchall()) == 0: # use len(cur.fetchall()) instead of cur.rouwcont in SQLite3
                print(f"Storing article {article['title']}.")
                cur.execute('insert into Queue(Title, Path) values (?, ?)', (article['title'], article['path']),)
                conn.commit()
            else:
                print(f"Article already exists: {article['title']}")

## define multitrhead function
visited = []
def get_links(thread_name, bs):
    print(f'Getting links in {thread_name}')
    links = bs.find('div',{'id':'bodyContent'}).find_all('a',href=re.compile('^(/wiki/)((?!:).)*$'))
    return [link for link in links if link not in visited]

def scrape_article(thread_name, path, queue):
    visited.append(path)
    html = requests.get(f'https://en.wikipedia.org{path}')
    time.sleep(5)
    bs = BeautifulSoup(html.text, 'html.parser')
    title = bs.find('h1').get_text()
    print(f'Added {title} for storing in thread {thread_name}')
    queue.put({'title':title,'path':path})
    links = get_links(thread_name, bs)
    if len(links) > 0:
        newArticle = links[random.randint(0, len(links) - 1)].attrs['href']
        scrape_article(thread_name, newArticle, queue)

queue = Queue()
try:
    _thread.start_new_thread(scrape_article,('Thread 1', '/wiki/Kevin_Bacon', queue,))
    _thread.start_new_thread(scrape_article,('Thread 2', '/wiki/Monty_Python', queue,))
    _thread.start_new_thread(storage, (queue,))
except:
    print('Error: unable to start threads')

while 1:
    pass

# Threading Model
import threading

## Using threading model instead of _+thread
threading.Thread(target=print_time, args=('Fizz',3,33)).start()
threading.Thread(target=print_time, args=('Buzz',5,20)).start()
threading.Thread(target=print_time, args=('Counter',1,100)).start()

def crawler(url):
    data = threading.local()
    data.visited = []

threading.Thread(target=crawler, args=('https://brookings.edu',)).start()

# Monitoring threading
class Crawler(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.done = False

    def isDone(self):
        return self.done

    def run(self):
        time.sleep(5)
        self.done = True
        raise Exception('Something bad happened!')

t = Crawler()
t.start()

while True:
    time.sleep(1)
    if t.isDone():
        print('Done~!')
        break
    if not t.isAlive():
        t = Crawler()
        t.start()

# Multiprocess Crawling

'''
## Cannot run the script since packages problem
from multiprocessing import Process

def print_time(threadName, delay, iterations):
    start = int(time.time())
    for i in range(0, iterations):
        time.sleep(delay)
        seconds_elapsed = str(int(time.time()) - start)
        print(threadName if threadName else seconds_elapsed)

processes = []
processes.append(Process(target=print_time, args=(None, 1, 100,)))
processes.append(Process(target=print_time, args=('Fizz', 3, 33,)))
processes.append(Process(target=print_time, args=('Buzz', 5, 20,)))

for p in processes:
    p.start()
for p in processes:
    p.join()
'''

## The following code cannot produce either with the same question above
'''
import os

def get_links(bs):
    print(f'Getting links in {os.getpid()}')
    links = bs.find('div',{'id':'bodyContent'}).find_all('a',href=re.compile('^(/wiki/)((?!:).)*$'))
    return [link for link in links if link not in visited]

def scrape_article(path):
    visited.append(path)
    html = requests.get(f'https://en.wikipedia.org{path}')
    time.sleep(5)
    bs = BeautifulSoup(html.text, 'html.parser')
    title = bs.find('h1').get_text()
    print(f'Added {title} for storing in thread {os.getpid()}')
    links = get_links(bs)
    if len(links) > 0:
        newArticle = links[random.randint(0, len(links) - 1)].attrs['href']
        print(newArticle)
        scrape_article(newArticle)

processes = []
processes.append(Process(target=scrape_article, args=('/wiki/Kevin_Bacon',)))
processes.append(Process(target=print_time, args=('/wiki/Monty_Python',)))

for p in processes:
    p.start()
'''

'''
With some question, I am not be able to produce multiprocess since the Process from 'multiprocess' is 
not working with the script I wrote.
'''
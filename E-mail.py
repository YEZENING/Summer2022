#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   May 24 16:13:18 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Summer2022
# @File:     E-mail.py
# @Software: PyCharm
"""
# Import Packages
import smtplib
from email.mime.text import MIMEText

msg = MIMEText('This is the first E-Mail send by Python! ZY')

msg['Subject'] = 'Hello World!'
msg['From'] = 'test@scraping.com'
msg['To'] = '1042638024@qq.com'

s = smtplib.SMTP('localhost')
s.send_message(msg)
s.quit()
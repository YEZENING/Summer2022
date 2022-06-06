#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on   Jun 02 13:58:23 2022
# @Author:   Zening Ye
# @Email:    zening.ye@gmail.com
# @Project:  Web Scrape
# @File:     Image Processing and Text Recognition.py
# @Software: PyCharm
"""
# Import Packages
import os, sys
from PIL import Image, ImageFilter


# PIL library
## Add　Blur into a image
cover = Image.open('Cover 3.jpeg')
print(cover.format, cover.size, cover.mode)
cover_blur = cover.filter(ImageFilter.GaussianBlur)
cover_blur.save('cover_blur.jpg')
cover_blur.show()

## Covert file into image
for infile in sys.argv[1:]:
    f, e = os.path.splitext(infile)
    outfile = f + '.jpg'
    if infile != outfile:
        try:
            with Image.open(infile) as im:
                im.save(outfile)
        except OSError:
            print('Cannot Convert', infile)
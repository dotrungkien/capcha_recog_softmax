from pwn import *

import socket
import datetime
import time
import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import random
import string
from preprocess import do_magic

def get_value(i, j, m):
    if m[i][j] == "@":
        return 255
    if m[i][j] == " ":
        return 0
    if m[i][j] == "-":
        return 0

conn = remote("128.199.113.197", 1111)
data = conn.recvuntil("Are you ready? [Y/n]")
print data
conn.send("Y\n")
while True:
    try:
        data = conn.recvuntil(" =")
    except:
        import pdb; pdb.set_trace()
    data = data[:data.index('Captcha')-1]
    print data

    pixel_map = []


    for line in data.split("\n"):
        pixel_map.append(list(line.strip()))

    # print pixel_map
    h, w = len(pixel_map), len(pixel_map[0])

    img = Image.new('L', (w, h), "white")
    pixels = img.load() # create the pixel map
    print pixels
    # import pdb; pdb.set_trace()
    print img.size[1], img.size[0]
    for i in range(img.size[1]):    # for every col:
        # print i
        for j in range(img.size[0]):    # For every row
            # print j
            pixels[j,i] = get_value(i, j, pixel_map)

    # img = img.resize((w*5, h*5), PIL.Image.NEAREST)
    fname = str(int(time.mktime(datetime.datetime.now().timetuple()))) + "-" + "".join([random.choice(string.lowercase) for x in range(5)]) + '.tif'
    print fname
    img.save("raw/" + fname)
    ans = do_magic("raw/" + fname)
    print ans
    conn.send(ans)

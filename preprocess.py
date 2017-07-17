from PIL import Image, ImageChops
from operator import itemgetter

import datetime
import time
import random
import string
import text_recog_logistic

# text_recog_logistic.predict('M-rdrfn.png')

def do_magic(file_name):
    captcha_filtered = Image.open(file_name)
    captcha_filtered = captcha_filtered.convert("P")
    inletter = False
    foundletter = False
    start = 0
    end = 0

    letters = []

    for y in range(captcha_filtered.size[0]): # slice across
        for x in range(captcha_filtered.size[1]): # slice down
            pix = captcha_filtered.getpixel((y,x))
            if pix == 0:
                inletter = True

        if foundletter == False and inletter == True:
            foundletter = True
            start = y

        if foundletter == True and inletter == False:
            foundletter = False
            end = y
            letters.append((start,end))

        inletter = False

    print letters
    all_length = letters[-1][1] - letters[0][0]
    print all_length

    new_letters = []
    if len(letters) != 5:
        min_length = min([b - a for a, b in letters])
        print min_length
        for a, b in letters:
            if b - a > int((all_length)/5 * 1.5):
                c = (b + a)/2
                new_letters.append((a, c))
                new_letters.append((c, b))
            else:
                new_letters.append((a, b))
    else:
        new_letters = letters
    print new_letters
    ans = ""
    for i in range(5):
        letter = new_letters[i]
        im3 = captcha_filtered.crop(( letter[0], 0, letter[1],captcha_filtered.size[1] ))
        im3.convert('L')

        # im3.save(str(i) + ".png")

        top = []
        down = []
        for y in range(im3.size[0]): # slice across
            for x in range(im3.size[1]): # slice down
                pix = im3.getpixel((y,x))
                # print pix
                if pix != 255:
                    top.append(x)
                    break
            for x in range(im3.size[1])[::-1]: # slice down
                pix = im3.getpixel((y,x))
                if pix != 255:
                    down.append(x)
                    break
        t, d = min(top), max(down)
        im3 = captcha_filtered.crop(( letter[0], t, letter[1], d)).resize((32,32))
        fn = "".join([random.choice(string.lowercase) for x in range(5)])
        img_path = 'data/samples/' + fn + ".png"
        im3.save(img_path)
        # print img_path, text_recog_logistic.predict(img_path)
        ans += text_recog_logistic.predict(img_path)

        # class Fit:
        #     letter = None
        #     difference = 0

        # best = Fit()

        # file_names = "MEPWNCTF"
        # for letter in file_names:
        #     #print letter
        #     current = Fit()
        #     current.letter = letter

        #     sample_path = "chars/" + letter + ".png"
        #     #print sample_path
        #     sample = Image.open(sample_path).convert('L').resize(im3.size)
        #     difference = ImageChops.difference(im3, sample)

        #     for x in range(difference.size[0]):
        #         for y in range(difference.size[1]):
        #             current.difference += difference.getpixel((x, y))

        #     if not best.letter or best.difference > current.difference:
        #         best = current

        # ans += best.letter
        # im4 = im3.resize((32, 32))
        # im4.save("samples_test/" + best.letter + "-" + "".join([random.choice(string.lowercase) for x in range(5)]) + ".png")

    print ans
    return ans

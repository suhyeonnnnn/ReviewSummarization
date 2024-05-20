from urllib import request
import time
from io import BytesIO
from PIL import Image
import json
import matplotlib.pyplot as plt
import os


def load_img(url, time_check = False):
    #url = "https://m.media-amazon.com/images/I/61OkTrsW88L._AC_UL1500_.jpg"

    # time check
    
    
    start = time.time()

    # request.urlopen()
    res = request.urlopen(url).read()

    # 이미지 다운로드 시간 체크
    if time_check == True:
        print(time.time() - start)


    # Image open
    img = Image.open(BytesIO(res))
    
    return img
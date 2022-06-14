import csv
import os

import numpy as np
from PIL import Image, ImageOps

counter = [0, 0, 0, 0]


class DataSet(object):

    def __init__(self, imagepath, tagfile, image_wsize, image_hsize):

        self._images = []
        self._tags = []

        with open(tagfile, 'r') as f:
            reader = csv.reader(f, delimiter=',')

            for idx, line in enumerate(reader):
                if idx > 0:
                    check = False
                    if float(line[5]) == 1.0 and float(line[25]) == 1.0 and counter[0] < 2033:
                        counter[0] += 1
                        check = True
                    elif float(line[5]) == 1.0 and float(line[25]) == -1.0 and counter[1] < 2033:
                        counter[1] += 1
                        check = True
                    elif float(line[5]) == -1.0 and float(line[25]) == 1.0 and counter[2] < 2033:
                        counter[2] += 1
                        check = True
                    elif float(line[5]) == -1.0 and float(line[25]) == -1.0 and counter[3] < 2033:
                        counter[3] += 1
                        check = True
                    if check:
                        filename = os.path.join(imagepath, line[0])
                        im1 = Image.open(filename)
                        img = ImageOps.fit(im1, size=(image_wsize, image_hsize), method=Image.LANCZOS,
                                           centering=(0.5, 0.5)).convert('RGB')
                        img = np.array(img, dtype=np.uint8)
                        im1.close()
                        self._images.append((img.astype(np.float32) / 127.5) - 1)
                        self._tags.append([line[5], line[21], line[25]])
                    if counter[0] == 2033 and counter[1] == 2033 and counter[2] == 2033 and counter[3] == 2033:
                        break

        self._images = np.array(self._images)
        self._tags = np.array(self._tags).astype(np.float32)
        self._image_num = len(self._tags)
        self._index_in_epoch = 0
        self.N_epoch = 0
        rnd = np.arange(0, self._image_num)
        np.random.shuffle(rnd)
        self._images = self._images[rnd]
        self._tags = self._tags[rnd]

        return

    def next_batch(self, batch_size=1):

        read_images = []
        caption = []

        for _ in range(batch_size):

            if self._index_in_epoch >= self._image_num:
                random_idx = np.arange(0, self._image_num)
                np.random.shuffle(random_idx)

                self._images = self._images[random_idx]
                self._tags = self._tags[random_idx]
                self._index_in_epoch = 0
                self.N_epoch += 1

            read_images.append(self._images[self._index_in_epoch])
            caption.append(self._tags[self._index_in_epoch])
            self._index_in_epoch += 1

        return read_images, np.array(caption)

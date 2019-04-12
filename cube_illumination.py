import cv2
import numpy as np

from roi_select import RoiSelector


def main():
    img = cv2.imread('/home/kkoscevic/zveri/zver3/data/cube/png/0002.png', -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img -= 2048
    img[img < 0] = 0
    img[img > img.max() - 2] = 0

    working_img = img.copy().astype(np.float32)
    working_img = (working_img / working_img.max() * 255).astype(np.uint8)

    mask = RoiSelector(working_img, 3).run()

    masked_img = np.ma.array(img, mask=mask)
    m = np.ma.median(masked_img, axis=(0, 1))
    m = m / m.sum()
    print(m)

    grey = np.array([[119] * 3])
    grey = grey / grey.sum()

    mm = m / grey
    print(mm / mm.sum())


if __name__ == '__main__':
    main()

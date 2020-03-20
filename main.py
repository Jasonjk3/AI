import cv2
import numpy as np
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True,
	help="path to input image")
parser.add_argument("-o", "--output", required=True,
	help="path to output image")
args = vars(parser.parse_args())

def img_blur_detection(imag):
    '''
    模糊检测
    :param imag:
    :return:
    '''
    if imag.all()==None:
        print('load image error!')
        return None
    grayImag = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    lapla = cv2.Laplacian(grayImag, cv2.CV_8U)
    imageVar = lapla.var()
    print('Laplacian值:',int(imageVar))
    if imageVar>500:
        return False
    else:
        return True

def img_exposure_detection(image):
    '''
    过曝检测
    :param image:
    :return:
    '''
    if image.all()==None:
        print('load image error!')
        return None
    def sliding_window(image, stepSize, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize[1]):
            for x in range(0, image.shape[1], stepSize[0]):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


    # 返回滑动窗结果集合，本示例暂时未用到
    def get_slice(image, stepSize, windowSize):
        slice_sets = []
        for (x, y, window) in sliding_window(image, stepSize, windowSize):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
                continue
            slice = image[y:y + windowSize[1], x:x + windowSize[0]]
            slice_sets.append(slice)
        return slice_sets

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = image[:, :, 2]
    # # 自定义滑动窗口的大小
    w = image.shape[1]
    h = image.shape[0]
    # # 本代码将图片分为3×3，共九个子区域，winW, winH和stepSize可自行更改
    (winW, winH) = (int(w / 27), int(h / 27))
    stepSize = (winW, winH)
    # (winW, winH)=(32,32)
    # stepSize=(32,32)
    zone = []
    count = 0
    for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # since we do not have a classifier, we'll just draw the window
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)


        slice = image[y:y + winH, x:x + winW]
        mean = np.mean(slice)
        if mean > 180:
            zone.append(slice)
        count += 1
    exposure_num = len(zone)
    result=exposure_num/count
    print('过曝占比值:',result)
    if result>0.1:
        return True
    else:
        return False

def img_exposure_detection_dark(image):
    '''
    偏暗检测
    :param image:
    :return:
    '''
    if image.all()==None:
        print('load image error!')
        return None
    def sliding_window(image, stepSize, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize[1]):
            for x in range(0, image.shape[1], stepSize[0]):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


    # 返回滑动窗结果集合，本示例暂时未用到
    def get_slice(image, stepSize, windowSize):
        slice_sets = []
        for (x, y, window) in sliding_window(image, stepSize, windowSize):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
                continue
            slice = image[y:y + windowSize[1], x:x + windowSize[0]]
            slice_sets.append(slice)
        return slice_sets

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = image[:, :, 2]
    # # 自定义滑动窗口的大小
    w = image.shape[1]
    h = image.shape[0]
    # # 本代码将图片分为3×3，共九个子区域，winW, winH和stepSize可自行更改
    (winW, winH) = (int(w / 27), int(h / 27))
    stepSize = (winW, winH)
    # (winW, winH)=(32,32)
    # stepSize=(32,32)
    zone = []
    count = 0
    for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # since we do not have a classifier, we'll just draw the window
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)


        slice = image[y:y + winH, x:x + winW]
        mean = np.mean(slice)
        if mean < 50:
            zone.append(slice)
        count += 1
    exposure_num = len(zone)
    result=exposure_num/count
    print('偏暗占比值:',result)
    if result>0.9:
        return True
    else:
        return False

def img_colorcast_detection(img):
    '''
    偏色检测
    :param img:
    :return:
    '''
    m, n, z = img.shape
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # print(img)
    l, a, b = cv2.split(img_lab)
    d_a, d_b, M_a, M_b = 0, 0, 0, 0

    d_a = a.sum() / (m * n) - 128
    d_b = b.sum() / (m * n) - 128

    D = np.sqrt((np.square(d_a) + np.square(d_b)))

    for i in range(m):
        for j in range(n):
            M_a = np.abs(a[i][j] - d_a - 128) + M_a
            M_b = np.abs(b[i][j] - d_b - 128) + M_b

    M_a, M_b = M_a / (m * n), M_b / (m * n)
    M = np.sqrt((np.square(M_a) + np.square(M_b)))
    k = D / M
    print('偏色值:%f' % k)
    if k>0.8:
        return True
    else:
        return False


if __name__ == '__main__':
    # img_paths=glob.glob(r'D:/临时/实训素材/test/*.jpg')
    img_paths=glob.glob(args["input"])

    if len(img_paths)==0:
        print('load images error!')
    output_path=args["output"]
    if os.path.exists(output_path)==False:
        os.mkdir(output_path)
    for index,path in enumerate(img_paths):
        print('图像正在处理(%i/%i)'%(index+1,len(img_paths)))
        img_name = path.split('\\')[-1]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        img_src=img.copy()
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w / 4), int(h / 4)), interpolation=cv2.INTER_CUBIC)
        print('输入图片->',img_name)
        print('图像像素->',img.shape)
        if img_blur_detection(img):#Ture 为图像模糊
            output_path_blur=output_path+'/模糊/'
            if os.path.exists(output_path_blur) == False:
                os.mkdir(output_path_blur)
            cv2.imencode('.jpg', img_src)[1].tofile(output_path_blur + img_name)
            print('模糊图像,输出路径->',output_path_blur + img_name)
        # else:#图像正常
        #     output_path_normal = output_path + '/正常/'
        #     if os.path.exists(output_path_normal) == False:
        #         os.mkdir(output_path_normal)
        #     cv2.imencode('.jpg', img)[1].tofile(output_path_normal + img_name)
        #     print('正常图像,输出路径->',output_path_normal + img_name)
        if img_exposure_detection(img):  # Ture 为图像过曝
            output_path_exposure = output_path + '/过曝/'
            if os.path.exists(output_path_exposure) == False:
                os.mkdir(output_path_exposure)
            cv2.imencode('.jpg', img_src)[1].tofile(output_path_exposure + img_name)
            print('过曝图像,输出路径->', output_path_exposure + img_name)

        if img_exposure_detection_dark(img):  # Ture 为图像偏暗
            output_path_exposure = output_path + '/偏暗/'
            if os.path.exists(output_path_exposure) == False:
                os.mkdir(output_path_exposure)
            cv2.imencode('.jpg', img_src)[1].tofile(output_path_exposure + img_name)
            print('偏暗图像,输出路径->', output_path_exposure + img_name)

        if img_colorcast_detection(img):  # Ture 为图像偏色
            output_path_colorcast = output_path + '/偏色/'
            if os.path.exists(output_path_colorcast) == False:
                os.mkdir(output_path_colorcast)
            cv2.imencode('.jpg', img_src)[1].tofile(output_path_colorcast + img_name)
            print('偏色图像,输出路径->', output_path_colorcast + img_name)
        print()
    print('Done')
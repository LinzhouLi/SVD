import numpy as np
from cv2 import cv2
import ImgSVDtransform as svd

if __name__ == '__main__':
    npz_path = input("请输入压缩文件路径: ")
    data = np.load(npz_path)
    r = svd.USD(data['Ur'], data['Sr'], data['Dr'])
    g = svd.USD(data['Ug'], data['Sg'], data['Dg'])
    b = svd.USD(data['Ub'], data['Sb'], data['Db'])
    img = np.stack((r, g, b), axis = 2)

    cv2.imshow("image after decompression", img)
    print("显示图片,按任意键结束...")
    cv2.waitKey(0)#防止图片一闪而过，等待键盘输入
    cv2.destroyAllWindows()
    c = input("是否需要保存图片?(y/n)")
    if c == "y" or c == "Y":
        name = input("请输入需要保存的文件名:")
        cv2.imwrite("%s.jpg" %(name), img, [int( cv2.IMWRITE_JPEG_QUALITY), 100])#以jpg格式最高质量保存图片

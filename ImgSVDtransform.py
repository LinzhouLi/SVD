import numpy as np
from cv2 import cv2

#对图片进行svd分解，返回3个色彩通道的U,S,D矩阵值
def image_svd_decompose(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    Ur,Sr,Dr = np.linalg.svd(r)
    Ug,Sg,Dg = np.linalg.svd(g)
    Ub,Sb,Db = np.linalg.svd(b)
    return Ur,Sr,Dr,Ug,Sg,Dg,Ub,Sb,Db

#对图片单一色彩通道的U,S,D矩阵相乘，取S的第m到第n项，返回单色彩通道svd处理后的图片矩阵
def USD(U, sigma, D, m = 0, n = 0):
    if n == 0:
        n = sigma.shape[0]
    S = np.zeros((U.shape[1], D.shape[0]))
    for i in range(m, n):
        S[i,i] = sigma[i]
    U_S = np.matmul(U, S)
    img_s_svd = np.matmul(U_S, D)
    img_s_svd[img_s_svd > 255] = 255
    img_s_svd[img_s_svd < 0] = 0
    img_s_svd = img_s_svd.astype(np.uint8)
    return img_s_svd

#对图片进行svd变换，取第m到第n项，返回处理后的图片矩阵
#m,n取值方式与range()函数相同
def image_svd_transform(img, m, n):
    Ur,Sr,Dr,Ug,Sg,Dg,Ub,Sb,Db = image_svd_decompose(img)
    r_svd = USD(Ur, Sr, Dr, m, n)
    g_svd = USD(Ug, Sg, Dg, m, n)
    b_svd = USD(Ub, Sb, Db, m, n)
    img_svd = np.stack((r_svd, g_svd, b_svd), axis = 2)
    return img_svd

if __name__ == '__main__':
    img_path = input("请输入图片路径: ")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    max = min(img.shape[0], img.shape[1])
    n = int(input("输入奇异值项数(1~%d): "%(max)))
    img_svd = image_svd_transform(img, 0, n)
    cv2.imshow("image after SVD transformation", img_svd)
    print("显示图片,按任意键结束...")
    cv2.waitKey(0)#防止图片一闪而过，等待键盘输入
    cv2.destroyAllWindows()
    c = input("是否需要保存图片?(y/n)")
    if c == "y" or c == "Y":
        name = input("请输入需要保存的文件名:")
        cv2.imwrite("%s.jpg" %(name), img_svd, [int( cv2.IMWRITE_JPEG_QUALITY), 100])#以jpg格式最高质量保存图片

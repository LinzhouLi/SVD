import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping

#图片单色彩通道进行矩阵SVD处理，取前n项
def image_svd_single_color(img_s,n):
    U,sigma,D = np.linalg.svd(img_s)
    S = np.zeros((U.shape[0], D.shape[1]))
    for i in range(n):
        S[i,i] = sigma[i]
    U_S = np.matmul(U, S)
    img_s_svd = np.matmul(U_S, D)
    img_s_svd[img_s_svd > 255] = 255
    img_s_svd[img_s_svd < 0] = 0
    img_s_svd = img_s_svd.astype(np.uint8)
    return img_s_svd

#图片SVD分解
def image_svd(img, n):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    r_svd = image_svd_single_color(r, n)
    g_svd = image_svd_single_color(g, n)
    b_svd = image_svd_single_color(b, n)
    img_svd = np.stack((r_svd, g_svd, b_svd), axis = 2)
    return img_svd

if __name__ == '__main__':
    img_path = input("请输入图片路径: ")
    img = mping.imread(img_path)
    max = min(img.shape[0], img.shape[1])
    n = int(input("输入奇异值项数(1~%d): "%(max)))
    pic = image_svd(img, n)
    plt.imshow(pic)
    plt.show()

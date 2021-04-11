import numpy as np
from cv2 import cv2
import ImgSVDtransform as svd

def count(S, percision):
    total = sum(S)
    temp = 0.0
    for i in range(S.shape[0]):
            temp += S[i]
            if temp / total >= percision:
                return i + 1
    return -1

def sigma(Sr, Sg, Sb, percision):
    sigma_r = count(Sr, percision)
    sigma_g = count(Sg, percision)
    sigma_b = count(Sb, percision)
    if sigma_r < 0 or sigma_g < 0 or sigma_b < 0:
        return -1
    return max(sigma_r, sigma_g, sigma_b)

def img_compression(img, percision):
    Ur,Sr,Dr,Ug,Sg,Dg,Ub,Sb,Db = svd.image_svd_decompose(img)
    print(Sr.shape[0])
    s = sigma(Sr, Sg, Sb, percision)
    print(s)
    Ur = Ur[:,:s]
    Ug = Ub[:,:s]
    Ub = Ub[:,:s]
    Sr = Sr[:s]
    Sg = Sg[:s]
    Sb = Sb[:s]
    Dr = Dr[:s,:]
    Dg = Db[:s,:]
    Db = Db[:s,:]
    return Ur,Sr,Dr,Ug,Sg,Dg,Ub,Sb,Db

if __name__ == '__main__':
    img_path = input("请输入图片路径: ")
    img = cv2.imread(img_path)
    percision = float(input("请输入压缩精度(0~1): "))
    Ur,Sr,Dr,Ug,Sg,Dg,Ub,Sb,Db = img_compression(img, percision)
    npz_path = img_path[:-4] + ".npz"
    np.savez(npz_path, Ur=Ur,Sr=Sr,Dr=Dr, Ug=Ug,Sg=Sg,Dg=Dg, Ub=Ub,Sb=Sb,Db=Db)
    

import cv2 as cv
import numpy as np
import argparse
import ctypes
import os
import glob
#from ctypes import *

debug_print_tmp = 1

version_suffix = "addn"
_K_ = 0.00074424
_A_ = 0.00000154
_B_ = 0.00161675
    
#I : 0~1
def get_noise_map(I,iso,bl):
    #noise model

    i8 = I*255
    
    K = _K_ * iso 
    B = _A_*iso*iso + _B_
    
    noise_map = K*(i8 - bl) + B
    
    noise_map = noise_map/255.0/255.0  #match the image data range 0~1
    
    return noise_map


def addNoise(I,iso,bl):

    i8 = I*255
    
    K = _K_ * iso 
    B = _A_*iso*iso + _B_
    
    sigma2 = K*(i8 - bl) + B
    
    mu = np.zeros_like(I)
    std = np.sqrt(sigma2)
    noise = np.random.normal (mu, std)
    
    I_addn = I + noise
    
    noise_map = K*(I_addn - bl) + B
    
    noise_map = noise_map/255.0/255.0  #match the image data range 0~1
    return I_addn,noise_map


def calcGST(inputIMG, w):
    img = inputIMG.astype(np.float32)
    # GST components calculation (start)
    # J =  (J11 J12; J12 J22) - GST
    imgDiffX = cv.Sobel(img, cv.CV_32F, 1, 0, 3)
    imgDiffY = cv.Sobel(img, cv.CV_32F, 0, 1, 3)
    imgDiffXY = cv.multiply(imgDiffX, imgDiffY)
    
    imgDiffXX = cv.multiply(imgDiffX, imgDiffX)
    imgDiffYY = cv.multiply(imgDiffY, imgDiffY)
    J11 = cv.boxFilter(imgDiffXX, cv.CV_32F, (w,w))
    J22 = cv.boxFilter(imgDiffYY, cv.CV_32F, (w,w))
    J12 = cv.boxFilter(imgDiffXY, cv.CV_32F, (w,w))
    # GST components calculations (stop)
    # eigenvalue calculation (start)
    # lambda1 = 0.5*(J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2))
    # lambda2 = 0.5*(J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2))
    tmp1 = J11 + J22
    tmp2 = J11 - J22
    tmp2 = cv.multiply(tmp2, tmp2)
    tmp3 = cv.multiply(J12, J12)
    tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
    lambda1 = 0.5*(tmp1 + tmp4)    # biggest eigenvalue
    lambda2 = 0.5*(tmp1 - tmp4)    # smallest eigenvalue
    # eigenvalue calculation (stop)
    # Coherency calculation (start)
    # Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
    # Coherency is anisotropy degree (consistency of local orientation)
    imgCoherencyOut = cv.divide(lambda1 - lambda2, lambda1 + lambda2)
    # Coherency calculation (stop)
    # orientation angle calculation (start)
    # tan(2*Alpha) = 2*J12/(J22 - J11)
    # Alpha = 0.5 atan2(2*J12/(J22 - J11))
    imgOrientationOut = cv.phase(J22 - J11, 2.0 * J12, angleInDegrees = True)
    imgOrientationOut = 0.5 * imgOrientationOut
    # orientation angle calculation (stop)
    return imgCoherencyOut, imgOrientationOut
    
    
    
    
def rgb2raw(rgb):
    bayer = np.zeros_like(rgb[:,:,0])
    bayer[0::2,0::2] = rgb[0::2,0::2,0]
    bayer[0::2,1::2] = rgb[0::2,1::2,1]
    bayer[1::2,0::2] = rgb[1::2,0::2,1]
    bayer[1::2,1::2] = rgb[1::2,1::2,2]
    
    return bayer


def variance_calc(I,dia,sigma):
    I_blur = cv.GaussianBlur(I,(dia,dia),sigma)
    
    hf = I - I_blur
    
    k = np.ones((5,5))
    k = k/np.sum(k)
    
    I_sqr = cv.filter2D(hf**2,-1,k)
    I_mean = cv.filter2D(hf,-1,k)
    variance = I_sqr - I_mean*I_mean
    
    return variance

def variance_grgb_calc(Gr,Gb,dia,sigma):
    Gr_blur = cv.GaussianBlur(Gr,(dia,dia),sigma)
    Gb_blur = cv.GaussianBlur(Gb,(dia,dia),sigma)
    
    I_blur = (Gr_blur + Gr_blur)/2.0
    
    Gr_hf = Gr - I_blur
    Gb_hf = Gb - I_blur
    
    k = np.ones((5,5))
    k = k/np.sum(k)
    
    Gr_sqr = cv.filter2D(Gr_hf**2,-1,k)
    Gb_sqr = cv.filter2D(Gb_hf**2,-1,k)
    Gr_mean = cv.filter2D(Gr_hf,-1,k)
    Gb_mean = cv.filter2D(Gb_hf,-1,k)
    I_sqr = (Gr_sqr+Gb_sqr)/2.0
    I_mean = (Gr_mean+Gb_mean)/2.0
    variance = I_sqr - I_mean*I_mean
    
    return variance


def grayRegion_XG_5x5(bayer):

    #color diffrence： calc the difference between G and r/b  ,and calc the difference between  r and b
    #do not calc between gr and gb
    expandw = 2
    bayer_exp = cv.copyMakeBorder(bayer,expandw,expandw,expandw,expandw,cv.BORDER_REFLECT_101)
    h,w = np.shape(bayer)
    bayer_exp = np.expand_dims(bayer_exp,axis = -1)
    imgs_G = np.zeros((h,w,12))
    imgs_X = np.zeros((h,w,4))
    
    #in 3x3 region
    imgs_X[:,:,0:1] = bayer_exp[1:h+1,1:w+1,0:1]
    imgs_G[:,:,0:1] = bayer_exp[1:h+1,2:w+2,0:1]
    imgs_X[:,:,1:2] = bayer_exp[1:h+1,3:w+3,0:1]
    imgs_G[:,:,1:2] = bayer_exp[2:h+2,1:w+1,0:1]
    imgs_G[:,:,2:3] = bayer_exp[2:h+2,3:w+3,0:1]
    imgs_X[:,:,2:3] = bayer_exp[3:h+3,1:w+1,0:1]
    imgs_G[:,:,3:4] = bayer_exp[3:h+3,2:w+2,0:1]
    imgs_X[:,:,3:4] = bayer_exp[3:h+3,3:w+3,0:1]
    
    #between 3x3 and 5x5 region
    imgs_G[:,:,4:5] = bayer_exp[0:h+0,1:w+1,0:1]
    imgs_G[:,:,5:6] = bayer_exp[0:h+0,3:w+3,0:1]
    imgs_G[:,:,6:7] = bayer_exp[1:h+1,0:w+0,0:1]
    imgs_G[:,:,7:8] = bayer_exp[1:h+1,4:w+4,0:1]
    imgs_G[:,:,8:9] = bayer_exp[3:h+3,0:w+0,0:1]
    imgs_G[:,:,9:10] = bayer_exp[3:h+3,4:w+4,0:1]
    imgs_G[:,:,10:11] = bayer_exp[4:h+4,1:w+1,0:1]
    imgs_G[:,:,11:12] = bayer_exp[4:h+4,3:w+3,0:1]
    
    bayer = np.expand_dims(bayer,axis = -1)
    diff_G = np.abs(bayer - imgs_G)
    diff_X = np.abs(bayer - imgs_X)
   
    min_diff_G = np.min(diff_G,axis = -1)
    min_diff_X = np.min(diff_X,axis = -1)
    
    min_diffg_r = min_diff_G[0::2,0::2]  #min diff between g and r
    min_diffg_b = min_diff_G[1::2,1::2]  #min diff between g and b
    color_diff = np.maximum(min_diffg_r,min_diffg_b)
    
    #min_diffx_r = min_diff_X[0::2,0::2]  #min diff between b and r
    #min_diffx_b = min_diff_X[1::2,1::2]  #min diff between b and r
    #color_diff_cc = np.maximum(color_diff_cg,min_diffx_b)
    
    #color_diff = np.maximum(color_diff_cg,color_diff_cc)
    
    
    min_diffg_gr = min_diff_X[0::2,1::2] #min diff between gb and gr
    min_diffg_gb = min_diff_X[1::2,0::2] #min diff between gb and gr
    min_diffg = np.minimum(min_diffg_gr,min_diffg_gb)
    
    print(np.shape(color_diff))
    #color_diff_blur = cv.GaussianBlur(color_diff,(7,7),1.0)
    color_diff_blur = cv.resize(color_diff,(w,h),interpolation = cv.INTER_LINEAR)
    min_diffg = cv.resize(min_diffg,(w,h),interpolation = cv.INTER_LINEAR)
    
    #local variance in g chn
    gr = bayer[0::2,1::2,0]
    gb = bayer[1::2,0::2,0]

    g_var = variance_grgb_calc(gr,gb,5,1.0)

    zeros_map = np.zeros_like(g_var)
    g_var = np.maximum(g_var,zeros_map)
    g_std = np.sqrt(g_var)
    g_std = cv.resize(g_std,(w,h),interpolation = cv.INTER_LINEAR)

    #gray region flag
    gray_ratio = color_diff_blur/(g_std + 0.000001)
    return color_diff_blur,g_std,min_diffg,gray_ratio
    
def grayRegion(bayer):

    #color diffrence： calc the difference between G and r/b  ,and calc the difference between  r and b
    #do not calc between gr and gb
    expandw = 1
    bayer_exp = cv.copyMakeBorder(bayer,expandw,expandw,expandw,expandw,cv.BORDER_REFLECT_101)
    h,w = np.shape(bayer)
    bayer_exp = np.expand_dims(bayer_exp,axis = -1)
    imgs_G = np.zeros((h,w,4))
    imgs_X = np.zeros((h,w,4))
    imgs_X[:,:,0:1] = bayer_exp[0:h,0:w,0:1]
    imgs_G[:,:,0:1] = bayer_exp[0:h,1:w+1,0:1]
    imgs_X[:,:,1:2] = bayer_exp[0:h,2:w+2,0:1]
    imgs_G[:,:,1:2] = bayer_exp[1:h+1,0:w+0,0:1]
    imgs_G[:,:,2:3] = bayer_exp[1:h+1,2:w+2,0:1]
    imgs_X[:,:,2:3] = bayer_exp[2:h+2,0:w+0,0:1]
    imgs_G[:,:,3:4] = bayer_exp[2:h+2,1:w+1,0:1]
    imgs_X[:,:,3:4] = bayer_exp[2:h+2,2:w+2,0:1]
    
    bayer = np.expand_dims(bayer,axis = -1)
    diff_G = np.abs(bayer - imgs_G)
    diff_X = np.abs(bayer - imgs_X)
   
    min_diff_G = np.min(diff_G,axis = -1)
    min_diff_X = np.min(diff_X,axis = -1)
    
    min_diffg_r = min_diff_G[0::2,0::2]  #min diff between g and r
    min_diffg_b = min_diff_G[1::2,1::2]  #min diff between g and b
    color_diff_cg = np.maximum(min_diffg_r,min_diffg_b)
    
    min_diffx_r = min_diff_X[0::2,0::2]  #min diff between b and r
    min_diffx_b = min_diff_X[1::2,1::2]  #min diff between b and r
    color_diff_cc = np.maximum(color_diff_cg,min_diffx_b)
    
    color_diff = np.maximum(color_diff_cg,color_diff_cc)
    
    
    min_diffg_gr = min_diff_X[0::2,1::2] #min diff between gr and gb
    min_diffg_gb = min_diff_X[1::2,0::2] #min diff between gr and gb
    min_diffg = np.minimum(min_diffg_gr,min_diffg_gb)
    
    print(np.shape(color_diff))
    #color_diff_blur = cv.GaussianBlur(color_diff,(7,7),1.0)
    color_diff_blur = cv.resize(color_diff,(w,h),interpolation = cv.INTER_LINEAR)
    min_diffg = cv.resize(min_diffg,(w,h),interpolation = cv.INTER_LINEAR)
    
    #local variance in g chn
    gr = bayer[0::2,1::2,0]
    gb = bayer[1::2,0::2,0]

    g_var = variance_grgb_calc(gr,gb,5,1.0)

    zeros_map = np.zeros_like(g_var)
    g_var = np.maximum(g_var,zeros_map)
    g_std = np.sqrt(g_var)
    g_std = cv.resize(g_std,(w,h),interpolation = cv.INTER_LINEAR)

    #gray region flag
    gray_ratio = color_diff_blur/(g_std + 0.000001)
    return color_diff_blur,g_std,min_diffg,gray_ratio
 
 
def grayRegion2(bayer):
    #color diffrence： calc the difference between G and r/b  ,and calc the difference between  r and b
    #do not calc between gr and gb
    expandw = 1
    bayer_exp = cv.copyMakeBorder(bayer,expandw,expandw,expandw,expandw,cv.BORDER_REFLECT_101)
    h,w = np.shape(bayer)
    bayer_exp = np.expand_dims(bayer_exp,axis = -1)
    imgs = np.zeros((h,w,8))
    
    imgs[:,:,0:1] = bayer_exp[0:h,0:w+0,0:1]
    imgs[:,:,1:2] = bayer_exp[0:h,1:w+1,0:1]
    imgs[:,:,2:3] = bayer_exp[0:h,2:w+2,0:1]
    imgs[:,:,3:4] = bayer_exp[1:h+1,0:w+0,0:1]
    imgs[:,:,4:5] = bayer_exp[1:h+1,2:w+2,0:1]
    imgs[:,:,5:6] = bayer_exp[2:h+2,0:w+0,0:1]
    imgs[:,:,6:7] = bayer_exp[2:h+2,1:w+1,0:1]
    imgs[:,:,7:8] = bayer_exp[2:h+2,2:w+2,0:1]
    
    bayer = np.expand_dims(bayer,axis = -1)
    diff = np.abs(bayer - imgs)
    
    min_diff = np.min(diff,axis = -1)
    
    min_diff_r = min_diff[0::2,0::2] #min diff between x and r
    min_diff_b = min_diff[1::2,1::2] #min diff between x and b
    color_diff = np.maximum(min_diff_r,min_diff_b)
    
    print(np.shape(color_diff))
    #color_diff_blur = cv.GaussianBlur(color_diff,(7,7),1.0)
    color_diff_blur = cv.resize(color_diff,(w,h),interpolation = cv.INTER_LINEAR)
    
    #local variance in g chn
    gr = bayer[0::2,1::2,0]
    gb = bayer[1::2,0::2,0]
    g_var = variance_grgb_calc(gr,gb,5,1.0)


    zeros_map = np.zeros_like(g_var)
    g_var = np.maximum(g_var,zeros_map)
    g_std = np.sqrt(g_var)
    g_std = cv.resize(g_std,(w,h),interpolation = cv.INTER_LINEAR)
    
    #gray region flag
    gray_ratio = color_diff_blur/(g_std + 0.000001)
    return color_diff_blur,g_std,gray_ratio
    
    

def ahd_fitFactor(bayer,details):
    #color diffrence： calc the difference between G and r/b  ,and calc the difference between  r and b
    #do not calc between gr and gb
    expandw = 2
    bayer_exp = cv.copyMakeBorder(bayer,expandw,expandw,expandw,expandw,cv.BORDER_REFLECT_101)
    h,w = np.shape(bayer)
    bayer_exp = np.expand_dims(bayer_exp,axis = -1)
    
    
    p0 = [0.15,0.0]
    p1 = [0.4 ,1.0]
    ones_map = np.ones_like(details)
    L  = p0[0]*ones_map
    H  = p1[0]*ones_map
    mask = np.where(details < L, L,details)
    mask = np.where(mask > H, H,mask)
    scale = p0[1] + (mask - p0[0]) * (p1[1] - p0[1])/(p1[0] - p0[0])
    scale = np.expand_dims(scale,axis = -1)
    
    ##horizental
    imgs = np.zeros((h,w,5))
    
    imgs[:,:,0:1] = bayer_exp[2:h+2,0:w+0,0:1]
    imgs[:,:,1:2] = bayer_exp[2:h+2,1:w+1,0:1]
    imgs[:,:,2:3] = bayer_exp[2:h+2,2:w+2,0:1]
    imgs[:,:,3:4] = bayer_exp[2:h+2,3:w+3,0:1]
    imgs[:,:,4:5] = bayer_exp[2:h+2,4:w+4,0:1]

    diff0 = np.abs(imgs[:,:,1:2] - imgs[:,:,3:4])/2.0
    diff1 = np.abs(2*imgs[:,:,2:3] - imgs[:,:,0:1] - imgs[:,:,4:5])/4.0
    h_fitF = diff0/(diff0 + diff1 + 0.000001)
    h_fitF = np.abs(h_fitF - 0.5)*scale
    
    
    ##vertical    
    imgs[:,:,0:1] = bayer_exp[0:h+0,2:w+2,0:1]
    imgs[:,:,1:2] = bayer_exp[1:h+1,2:w+2,0:1]
    imgs[:,:,2:3] = bayer_exp[2:h+2,2:w+2,0:1]
    imgs[:,:,3:4] = bayer_exp[3:h+3,2:w+2,0:1]
    imgs[:,:,4:5] = bayer_exp[4:h+4,2:w+2,0:1]

    diff0 = np.abs(imgs[:,:,1:2] - imgs[:,:,3:4])/2.0
    diff1 = np.abs(2*imgs[:,:,2:3] - imgs[:,:,0:1] - imgs[:,:,4:5])/4.0
    v_fitF = diff0/(diff0 + diff1 + 0.000001)
    v_fitF = np.abs(v_fitF - 0.5)*scale
    
    return h_fitF,v_fitF 
    
    
def kernel_regression_callC(img_src,lib):
    print("::kernel_regression_callC")
    h, w = img_src.shape
    expandw = 8
    stride = w + 2*expandw
    
    img_src_exp = cv.copyMakeBorder(img_src,expandw,expandw,expandw,expandw,cv.BORDER_REPLICATE)
    print("img_src_exp:",np.shape(img_src_exp),stride)
    h_exp, w_exp = img_src_exp.shape
    
    
    data_len = h*w
    data_exp_len = h_exp*w_exp
    
    img_src_flattern = img_src_exp.reshape(data_exp_len)
    
    #input
    IMG_EXP_BUFF = ctypes.c_float * data_exp_len
    c_img_src = IMG_EXP_BUFF()
    c_img_guide = IMG_EXP_BUFF()
    c_img_noisemap = IMG_EXP_BUFF()
    for i in range(data_exp_len):
        c_img_src[i] = img_src_flattern[i]
        c_img_guide[i] = img_src_flattern[i]
        c_img_noisemap[i] = img_src_flattern[i]
    
    #output data     
    IMG_BUFF = ctypes.c_float * data_len
    c_img_dst = IMG_BUFF()
    c_img_dir = IMG_BUFF()
    c_img_dirS = IMG_BUFF()
    c_img_edgeS = IMG_BUFF()
    c_img_kernel = IMG_BUFF()
    for i in range(data_len):
        c_img_dst[i] = 0
        c_img_dir[i] = 0
        c_img_dirS[i] = 0
        c_img_edgeS[i] = 0
        c_img_kernel[i] = 0
        
    gradient_w = 2
    radius_w = 2
    
    
    #nr_steerkernel( float *pSrc, 
    #                 float *pGuide, 
    #                float *pNoiseMap,
    #                 int srcStride,
    #                 float *pDst, 
    #                float *pDir, 
    #                float *pDirS, 
    #                float *pEdgeS, 
    #                float *pKernel, 
    #                 int dstStride, 
    #                 int width, 
    #                 int height,
    #                int expand_w,
    #                int gradient_w,
    #                 int radius_w
    #                 )
    
    lib.nr_steerkernel(c_img_src,c_img_guide,c_img_noisemap,ctypes.c_int(stride),c_img_dst,c_img_dir,c_img_dirS,c_img_edgeS,c_img_kernel,ctypes.c_int(w),ctypes.c_int(w),ctypes.c_int(h),ctypes.c_int(expandw),ctypes.c_int(gradient_w),ctypes.c_int(radius_w))
    #lib.test(c_img_noisemap,c_img_dir,ctypes.c_int(w_exp),ctypes.c_int(h_exp),ctypes.c_int(w),ctypes.c_int(h),ctypes.c_int(expandw))

    img_dst = np.zeros(data_len)
    img_dir = np.zeros(data_len)
    img_dirS = np.zeros(data_len)
    img_edgeS = np.zeros(data_len)
    img_kernel = np.zeros(data_len)
    
    for i in range(data_len):
        img_dst[i] = c_img_dst[i]
        img_dir[i] = c_img_dir[i]
        img_dirS[i] = c_img_dirS[i]
        img_edgeS[i] = c_img_edgeS[i]
        img_kernel[i] = c_img_kernel[i]
    
    img_dst = img_dst.reshape((h,w))
    img_dir = img_dir.reshape((h,w))
    img_dirS = img_dirS.reshape((h,w))
    img_edgeS = img_edgeS.reshape((h,w))
    img_kernel = img_kernel.reshape((h,w))
    
    tmp = np.clip(img_dst*255,0,255.0)
    cv.imwrite('./output/img_dst.bmp',tmp)
    
    tmp = np.clip(img_dir,0,255.0)
    cv.imwrite('./output/img_dir.bmp',tmp)
    
    tmp = np.clip(img_dirS*255,0,255.0)
    cv.imwrite('./output/img_dirS.bmp',tmp)
    
    tmp = np.clip(img_edgeS*255,0,255.0)
    cv.imwrite('./output/img_edgeS.bmp',tmp)
    
    tmp = np.clip(img_kernel*255,0,255.0)
    cv.imwrite('./output/img_kernel.bmp',tmp)
    


def medianF(img_src):
    print("::medianF")
    h, w = img_src.shape
    expandw = 1
    stride = w + 2*expandw
    
    img_src_exp = cv.copyMakeBorder(img_src,expandw,expandw,expandw,expandw,cv.BORDER_REPLICATE)
    h_exp, w_exp = img_src_exp.shape
    
    I0 = img_src_exp[0:h,0:w]
    I1 = img_src_exp[0:h,1:w+1]
    I2 = img_src_exp[0:h,2:w+2]
    
    I3 = img_src_exp[1:h+1,0:w]
    I4 = img_src_exp[1:h+1,1:w+1] #same as img_src
    I5 = img_src_exp[1:h+1,2:w+2]
    
    I6 = img_src_exp[2:h+2,0:w]
    I7 = img_src_exp[2:h+2,1:w+1]
    I8 = img_src_exp[2:h+2,2:w+2]
    
    #horizental
    med_h = (I3 + I4 + I5) - np.maximum(np.maximum(I3,I4),I5) - np.minimum(np.minimum(I3,I4),I5)
    med_v = (I1 + I4 + I7) - np.maximum(np.maximum(I1,I4),I7) - np.minimum(np.minimum(I1,I4),I7)
    
    median_v = (med_h + med_v)/2.0
    
    return median_v



def dm_ahd_callC(img_src,bayer,blr,lib,bayer_f,file_name):

    h, w,_ = img_src.shape

    
    data_len = h*w
    RGB_len = 3*h*w
    
    bayer_flattern = bayer.reshape(data_len)
    
    #input
    IMG_EXP_BUFF = ctypes.c_short * data_len
    c_bayer = IMG_EXP_BUFF()
    c_Gh = IMG_EXP_BUFF()
    c_Gv = IMG_EXP_BUFF()
    c_homo_h = IMG_EXP_BUFF()
    c_homo_v = IMG_EXP_BUFF()
    c_homo_bayer = IMG_EXP_BUFF()



    for i in range(data_len):
        c_bayer[i] = bayer_flattern[i]

    
    #output data     
    IMG_BUFF = ctypes.c_short * RGB_len
    c_img_dst = IMG_BUFF()
    

        
    gradient_w = 2
    radius_w = 2
    
    VH_blr_par0 = 400 #__linear_interr(iso,500,1200,400,128);
    VH_blr_par1 = 8
    
    #demosaic_ahd(short *pRaw,int width, int height, short *pRGB, short *pG_h, short *pG_v, short *pHomo_h, short *pHomo_v, int bit_depth, int VH_blr_par0, int VH_blr_par1)
    lib.demosaic_ahd(c_bayer,w, h, c_img_dst, c_Gh, c_Gv, c_homo_h, c_homo_v, bitw,  VH_blr_par0,  VH_blr_par1)
    #
    #demosaic_ahd_weighted is not robust
    #lib.demosaic_ahd_weighted(c_bayer,w, h, c_img_dst, c_Gh, c_Gv, c_homo_h, c_homo_v, bitw,  VH_blr_par0,  VH_blr_par1)
    
    #demosaic_ahd_blendwithbayer is not robust
    #lib.demosaic_ahd_blendwithbayer(c_bayer,w, h, c_img_dst, c_Gh, c_Gv, c_homo_h, c_homo_v,c_homo_bayer, bitw,  VH_blr_par0,  VH_blr_par1)
    
    
    
    img_dst = np.zeros(RGB_len)
    Gh = np.zeros(data_len)
    Gv = np.zeros(data_len)

    homo_h = np.zeros(data_len)
    homo_v = np.zeros(data_len)
    homo_bayer = np.zeros(data_len)

    
    for i in range(RGB_len):
        img_dst[i] = c_img_dst[i]
        
    for i in range(data_len):
        Gh[i] = c_Gh[i]
        Gv[i] = c_Gv[i]
        homo_h[i] = c_homo_h[i]
        homo_v[i] = c_homo_v[i]
        homo_bayer[i] = c_homo_bayer[i]


    
    img_dst = img_dst.reshape((3,h,w))
    img_dst = img_dst.transpose(1,2,0)
    
    Gh = Gh.reshape((h,w))
    Gv = Gv.reshape((h,w))

    homo_h = homo_h.reshape((h,w))
    homo_v = homo_v.reshape((h,w))
    homo_bayer = homo_bayer.reshape((h,w))

    
    img_dst = img_dst.astype(np.float32)
    img_dst = img_dst/(2**14-1)
    
    Gh = Gh.astype(np.float32)
    Gh = Gh/(2**14-1)
    
    Gv = Gv.astype(np.float32)
    Gv = Gv/(2**14-1)
    
    #print(img_dst[16,16:32,1])

    
    
    g = img_dst[:,:,1] * (1.0 - blr) + bayer_f*blr
    #print(g[16,16:32])
    
    
    
    if debug_print_tmp == 1:
        tmp = np.clip(g*print_img_gain,0,255.0)
        cv.imwrite('./output/%s_img_dst_g_blendwithBayer.bmp'%(file_name),tmp[:,:])
        
        tmp = np.clip(Gh*print_img_gain,0,255.0)
        cv.imwrite('./output/Gh.bmp',tmp)
        
        tmp = np.clip(Gv*print_img_gain,0,255.0)
        cv.imwrite('./output/Gv.bmp',tmp)
        
        
        tmp = np.clip(homo_h*4,0,255.0)
        cv.imwrite('./output/homo_h.bmp',tmp)
        
        tmp = np.clip(homo_v*4,0,255.0)
        cv.imwrite('./output/homo_v.bmp',tmp)
        
        tmp = np.clip(homo_bayer*4,0,255.0)
        cv.imwrite('./output/homo_bayer.bmp',tmp)
        
    return img_dst

def noise_model(iso):
    #raw luma-noise curve
    _K_ = 0.00074424
    _A_ = 0.00000154
    _B_ = 0.00161675
    
    b = _A_ * iso * iso + _B_ * iso;
    k = _K_
    return k,b


def add_noise(I,iso):
    tmp = I*255.0
    K,B = noise_model(iso)
    
    size = np.shape(tmp)
    mean = np.zeros_like(tmp)
    sigma = K * (tmp-16.0) + B
    sigma = np.maximum(sigma,0.0)
    std = np.sqrt(sigma)
    
    noise = np.random.normal(mean,std,size)
    
    I_addn = tmp + noise
    I_addn = I_addn/255.0
    I_addn = I_addn*1023 + 0.5
    I_addn = I_addn.astype(np.uint16)
    I_addn = I_addn.astype(np.float32)
    I_addn = I_addn/1023.0
    
    noise_map = K * (I_addn*255.0-16.0) + B
    noise_map = np.maximum(noise_map,0.0)
    
    noise_map = noise_map/255.0/255.0
    
    return I_addn,noise_map


def deGamma(I):
    I = I ** 3.0
    
    return I

    
#D:\1.data_set\vimeo_interp_test\target\00004\0706\
if __name__ == '__main__':
    iso = 3200
    bl = 1.0/16.0
    
    bl_print = 0.5/16.0
    
    
    
    ##############
    print_img_gain = 255
    select_file = ['Testphoto.jpg']#,'Color_Print_Test03.jpg']
    len_select = len(select_file)
    print("len_select:",len_select)
    file_path_root = '../../../high_quality_images/' 
    
    files = glob.glob(os.path.join(file_path_root,'*.jpg'))
    files_num = len(files)
    for file_idx in range(files_num):
        file_path = files[file_idx]
        
        #file_path = '../image/t1.jpg'
        file_name = os.path.basename(file_path)
        
        
        if len_select > 0:
            flag = file_name in  select_file
            if(flag == False):
                print(file_name," not in the list")
                continue
        
        print("run file_name :",file_name)
        file_name = file_name[0:-4]
        
        
        img = cv.imread(file_path)
        img = img.astype(np.float32)
        h,w,chn = np.shape(img)
        print(np.shape(img))
        
        h_ = (h//2)*2
        w_ = (w//2)*2
        
        img = img[0:h_,0:w_,:]
        print(np.shape(img))
        
        
        #
        ##########################################
        #
        lib = ctypes.CDLL(".\kr.dll")
        
        #
        ########input data
        img_src = np.asarray(img)
        img_src = img_src/255.0
        
        img_src = deGamma(img_src)
        
        img_src = img_src*(1.0-bl) + bl
        
        print("img_src:",np.shape(img_src))
        
        tmp = np.clip(img_src-bl_print,0.0,1.0)**(1.0/2.2)
        tmp = np.clip(tmp*print_img_gain,0,255.0)
        cv.imwrite('./output/%s_img_src.bmp'%(file_name),tmp)
        cv.imwrite('./output/%s_img_src_G.bmp'%(file_name),tmp[:,:,1])
        
        
        bayer_clean = rgb2raw(img_src)
         
        bayer,noise_map = add_noise(bayer_clean,iso)
        
        ##############################################DM
        bitw = 14
        bayer2Y = np.array([[1,2,1],
                            [2,4,2],
                            [1,2,1]])
        bayer2Y = bayer2Y/np.sum(bayer2Y)
        print(np.shape(bayer2Y))
        print(bayer2Y)
        y = cv.filter2D(bayer,-1,bayer2Y)
        
        if debug_print_tmp == 1:
            tmp = np.clip(y*print_img_gain,0,255.0)
            cv.imwrite('./output/%s_y.bmp'%(file_name),tmp)
            
            tmp = np.clip(bayer*print_img_gain,0,255.0)
            cv.imwrite('./output/%s_bayer.bmp'%(file_name),tmp)
            
            cv.imwrite('./output/%s_bayer_r.bmp'%(file_name),tmp[0::2,0::2])
            cv.imwrite('./output/%s_bayer_gr.bmp'%(file_name),tmp[0::2,1::2])
            cv.imwrite('./output/%s_bayer_gb.bmp'%(file_name),tmp[1::2,0::2])
            cv.imwrite('./output/%s_bayer_b.bmp'%(file_name),tmp[1::2,1::2])
        
        
        
        #gray high frequency reigon
        minDiffxg,_,minDiffg,gray_ratio = grayRegion(bayer)
        minDiff_blur2,g_std,gray_ratio2 = grayRegion2(bayer)
        
        
        #minDiffxg_5x5,_,minDiffg,_ = grayRegion_XG_5x5(bayer)
        
        
        p0 = [0.01,1.0]
        p1 = [0.12,0.0]
        ones_map = np.ones_like(g_std)
        L  = p0[0]*ones_map
        H  = p1[0]*ones_map
        mask = np.where(g_std < L, L,g_std)
        mask = np.where(mask > H, H,mask)
        gray_blr = p0[1] + (mask - p0[0]) * (p1[1] - p0[1])/(p1[0] - p0[0])
        
        #in fine high frequency detail region  , trust the difference with all surrounding pixels
        minDiff = minDiff_blur2 * (1-gray_blr) + minDiffxg*gray_blr
        

        gray_ratio3 = minDiff/(minDiffg+0.000001)

        p0 = [0.6,0.6]
        p1 = [3.0,0.0]
        ones_map = np.ones_like(gray_ratio3)
        L  = p0[0]*ones_map
        H  = p1[0]*ones_map
        mask = np.where(gray_ratio3 < L, L,gray_ratio3)
        mask = np.where(mask > H, H,mask)
        blr = p0[1] + (mask - p0[0]) * (p1[1] - p0[1])/(p1[0] - p0[0])

        if debug_print_tmp == 1:
            tmp = np.clip(blr*255,0,255.0)
            cv.imwrite('./output/%s_blr.bmp'%(file_name),tmp)
            
            
            tmp = np.clip(g_std*255,0,255.0)
            cv.imwrite('./output/%s_g_std.bmp'%(file_name),tmp)
            
            tmp = np.clip(minDiff*255*4,0,255.0)
            cv.imwrite('./output/%s_minDiff.bmp'%(file_name),tmp)
            
            #tmp = np.clip(minDiffxg*255*4,0,255.0)
            #cv.imwrite('./output/%s_minDiffxg.bmp'%(file_name),tmp)
            
            #tmp = np.clip(minDiffxg_5x5*255*4,0,255.0)
            #cv.imwrite('./output/%s_minDiffxg_5x5.bmp'%(file_name),tmp)
            
            tmp = np.clip(minDiffg*255*4,0,255.0)
            cv.imwrite('./output/%s_minDiffg.bmp'%(file_name),tmp)
            
            tmp = np.clip(minDiff_blur2*255*4,0,255.0)
            cv.imwrite('./output/%s_minDiff_blur2.bmp'%(file_name),tmp)
            
            #tmp = np.clip(gray_ratio*16,0,255.0)
            #cv.imwrite('./output/%s_gray_ratio.bmp'%(file_name),tmp)
            tmp = np.clip(gray_ratio2*16,0,255.0)
            cv.imwrite('./output/%s_gray_ratio2.bmp'%(file_name),tmp)
            tmp = np.clip(gray_ratio3*16,0,255.0)
            cv.imwrite('./output/%s_gray_ratio3.bmp'%(file_name),tmp)
        
        bayer_f = bayer.copy()
        
        
        h_fitF,v_fitF  = ahd_fitFactor(bayer_f,g_std)
         
        if debug_print_tmp == 1:
            tmp = np.clip(h_fitF*255,0,255.0)
            cv.imwrite('./output/%s_h_fitF.bmp'%(file_name),tmp)
            
            tmp = np.clip(v_fitF*255,0,255.0)
            cv.imwrite('./output/%s_v_fitF.bmp'%(file_name),tmp)
        
        #kernel_regression_callC(bayer_f,lib)
        
        bayer = bayer*(2**14-1)
        bayer = bayer.astype(np.int32)
        img_dst = dm_ahd_callC(img_src,bayer,blr,lib,bayer_f,file_name)

        tmp = np.clip(img_dst-bl_print,0.0,1.0)**(1.0/2.2)
        tmp = np.clip(tmp*print_img_gain,0,255.0)
        cv.imwrite('./output/%s_img_dst_%s.bmp'%(file_name,version_suffix),tmp)
        
        cv.imwrite('./output/%s_img_dst_%s_g.bmp'%(file_name,version_suffix),tmp[:,:,1])
        
        ##############################NR
        g_medianF = medianF(img_dst[:,:,1])
        
        tmp = np.clip(g_medianF-bl_print,0.0,1.0)**(1.0/2.2)
        tmp = np.clip(tmp*print_img_gain,0,255.0)
        cv.imwrite('./output/%s_img_dst_%s_g_medianF.bmp'%(file_name,version_suffix),tmp)
        
        
    # print(__name__)
    
    
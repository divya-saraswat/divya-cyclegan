import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def unnorm(img, mean, std):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(s)

    return img


def show_example(data_loaded):
    photo_img, monet_img = next(iter(data_loaded))

    f = plt.figure(figsize=(8, 8))

    f.add_subplot(1, 2, 1)
    plt.title('Photo')
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    photo_img = unnorm(photo_img, mean, std)
    plt.imshow(photo_img[0].permute(1, 2, 0))

    f.add_subplot(1, 2, 2)
    plt.title('Monet')
    monet_img = unnorm(monet_img, mean, std)
    plt.imshow(monet_img[0].permute(1, 2, 0))
    plt.grid(False)
    plt.show()


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


def estimate_transmission(src):
    I = src.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    # J = Recover(I,t,A,0.1)
    return t


def guidedFilter(I, p, r, eps):
    hei, wid = p.shape
    N = cv2.boxFilter(np.ones((hei, wid)), -1, (r,r))

    meanI = cv2.boxFilter(I, -1, (r,r)) / N
    meanP = cv2.boxFilter(p, -1, (r,r)) / N
    corrI = cv2.boxFilter(I * I, -1, (r,r)) / N
    corrIp = cv2.boxFilter(I * p, -1, (r,r)) / N

    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = cv2.boxFilter(a, -1, (r,r)) / N
    meanB = cv2.boxFilter(b, -1, (r,r)) / N

    q = meanA * I + meanB
    return q


def find_darkchannel(image, patch_win_size):
    patch_win_size = int(patch_win_size)
    b,g,r = cv2.split(image)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(patch_win_size,patch_win_size))
    dark_channel = cv2.erode(dc,kernel)
    return dark_channel


def find_atmosphericLight(image, dark_channel):

    m, n = image.shape[0], image.shape[1]

    search_A_pixel = np.floor(m*n*0.01)
    image_save = np.reshape(image, (m*n, 3))
    darkchannel_save = np.reshape(dark_channel, m*n)

    saver = np.zeros((1, 3))
    idx = np.argsort(-darkchannel_save)

    for pixel_idx in range(int(search_A_pixel)):
        saver = saver + image_save[ idx[pixel_idx], :]

    A = saver / search_A_pixel
    return A


def dehaze_patchMap(image, omega, patchMap):
    m, n = image.shape[0], image.shape[1]

    transmissionMap = np.ones((m, n))
    darkchannelMap = np.ones((m, n))

    patchMap = np.ceil(patchMap)
    # patchMap(find(patchMap<1))=1
    # patchMap(find(patchMap>120))=120
    patchMap[patchMap < 1] = 1
    patchMap[patchMap > 120] = 120

    '''patchMap = guided_filter(rgb2gray(image), patchMap, 15, 0.001)
    patchMap = ceil(patchMap)
    patchMap(find(patchMap<1))=1
    patchMap(find(patchMap>120))=120'''
    image = image[0, :, :, :].T
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255
    print(type(gray), gray.shape, type(patchMap), patchMap)
    patchMap = guidedFilter(gray, patchMap, 15, 0.001).astype(np.uint8)
    patchMap[patchMap < 1] = 1
    patchMap[patchMap > 120] = 120

    '''[patch_size, ~, patchIdx] = unique(patchMap)
    patchIdx = reshape(patchIdx, m, n)
    patch_size_num = size(patch_size)'''

    patch_size = np.unique(patchMap)

    '''for i = 1: patch_size_num(1)
        i
        dark_channel = find_darkchannel(image, patch_size(i))
        atmosphere = find_atmosphericLight(image, dark_channel)
        atmosphere_est = repmat(reshape(atmosphere, [1, 1, 3]), m, n)
        est_term = image./atmosphere_est
        tx_estimation = 1-omega*find_darkchannel(est_term, patch_size(i))
        tx_estimation = reshape(tx_estimation, m, n)
        patchIdx = patchMap == patch_size(i)
        transmissionMap(patchIdx) = tx_estimation(patchIdx)
        darkchannelMap(patchIdx) = dark_channel(patchIdx)     
        tx_refine = guided_filter(rgb2gray(image), transmissionMap, 15, 0.001)
        tx_refine = reshape(tx_refine, m, n)
        A_predict = find_atmosphericLight(image, darkchannelMap)
        A_predict = repmat(reshape(A_predict, [1, 1, 3]), m, n)
        tx = repmat(max(tx_refine, 0.1), [1, 1, 3])
        recover_result = ((image - A_predict) ./ tx) + A_predict'''

    image = image / 255
    for i in range(len(patch_size)):
        print(str(i), end=' ')
        dark_channel = find_darkchannel(image, patch_size[i])

        atmosphere = find_atmosphericLight(image, dark_channel)

        # atmosphere_est = repmat(reshape(atmosphere, [1, 1, 3]), m, n)
        # est_term = image./atmosphere_est

        '''atmosphere_est = np.zeros((m, n, 3))
        atmosphere_est[...,0] = atmosphere[0]
        atmosphere_est[...,1] = atmosphere[1]
        atmosphere_est[...,2] = atmosphere[2]
        est_term = image / atmosphere_est'''
        est_term = image / atmosphere

        tx_estimation = 1 - omega * find_darkchannel(est_term, patch_size[i])
        tx_estimation = np.reshape(tx_estimation, (m, n))
        '''patchIdx = patchMap == patch_size(i)
        transmissionMap(patchIdx) = tx_estimation(patchIdx)
        darkchannelMap(patchIdx) = dark_channel(patchIdx)  '''
        patchIdx = patchMap == patch_size[i]
        transmissionMap[patchIdx] = tx_estimation[patchIdx]
        darkchannelMap[patchIdx] = dark_channel[patchIdx]

        '''tx_refine = guided_filter(rgb2gray(image), transmissionMap, 15, 0.001)
        tx_refine = reshape(tx_refine, m, n)
        A_predict = find_atmosphericLight(image, darkchannelMap)
        A_predict = repmat(reshape(A_predict, [1, 1, 3]), m, n)
        tx = repmat(max(tx_refine, 0.1), [1, 1, 3])
        recover_result = ((image - A_predict) ./ tx) + A_predict'''
        tx_refine = guidedFilter(gray, transmissionMap, 15,
                                 0.001)  # guided_filter(rgb2gray(image), transmissionMap, 15, 0.001)
        # tx_refine = reshape(tx_refine, m, n)
        A_predict = find_atmosphericLight(image, darkchannelMap)
        # A_predict = repmat(reshape(A_predict, [1, 1, 3]), m, n)
        # tx = repmat(max(tx_refine, 0.1), [1, 1, 3])
        tx = np.reshape(tx_refine, (m, n, 1))
        tx[tx < 0.1] = 0.1
        tx = np.concatenate((tx, tx, tx), axis=2)
        recover_result = (image - A_predict) / tx + A_predict
        # plt.imshow(recover_result); plt.show()

    return recover_result * 255, tx
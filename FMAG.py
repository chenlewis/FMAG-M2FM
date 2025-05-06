import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


def RGB_fft(img):
    img_b = img[:, :, 0]
    f_b = np.fft.fft2(img_b)
    fshift_b = np.fft.fftshift(f_b)
    fshift_b_angel = np.angle(fshift_b)

    img_g = img[:, :, 1]
    f_g = np.fft.fft2(img_g)
    fshift_g = np.fft.fftshift(f_g)
    fshift_g_angel = np.angle(fshift_g)

    img_r = img[:, :, 2]
    f_r = np.fft.fft2(img_r)
    fshift_r = np.fft.fftshift(f_r)
    fshift_r_angel = np.angle(fshift_r)

    fshift_angel = np.stack((fshift_b_angel, fshift_g_angel, fshift_r_angel), -1)
    fshift_origin_abs = np.stack((np.abs(fshift_b), np.abs(fshift_g), np.abs(fshift_r)), -1)

    return fshift_origin_abs, fshift_angel


def RGB_ifft(img_fft):
    img_fft_b = img_fft[:, :, 0]
    img_fft_g = img_fft[:, :, 1]
    img_fft_r = img_fft[:, :, 2]
    img_aug_b = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft_b)))
    img_aug_g = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft_g)))
    img_aug_r = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft_r)))
    img_aug = np.stack((img_aug_b, img_aug_g, img_aug_r), -1).astype(np.uint8)
    return img_aug


def show(img):
    plt.figure(figsize=(12, 12))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    fshift_abs = np.log(np.abs(fshift))
    plt.imshow(fshift_abs, 'gray')
    plt.show()


def add_noise(spectral, mask, a1=0.5, a2=0.8, mu=0, sigma=1): 
    height, width, depth = spectral.shape
    mask1 = 1 - mask

    alpha = np.random.uniform(a1, a2, height * width * depth)
    alpha = alpha.reshape((height, width, depth))

    beta = np.random.normal(mu, sigma, height * width * depth)
    beta = beta.reshape((height, width, depth))

    # spectral_mask_noise = (alpha * spectral + beta) * mask + spectral * mask1
    spectral_mask_noise = (alpha * spectral + beta) * mask
    return spectral_mask_noise


def get_mask(img_fft, raw_peek):
    mask_raw = np.zeros((224, 224, 3), np.uint8)
    cv2.circle(mask_raw, (112 + raw_peek, 112), 15, (1, 1, 1), -1)

    img_fft_mask_raw = img_fft * mask_raw
    max_pos = np.argmax(img_fft_mask_raw)
    max_coord = np.unravel_index(max_pos, img.shape)
    peek_1 = max_coord[1] - 112

    mask_raw = np.zeros((224, 224, 3), np.uint8)
    cv2.circle(mask_raw, (112 - raw_peek, 112), 15, (1, 1, 1), -1)
    img_fft_mask_raw = img_fft * mask_raw
    max_pos = np.argmax(img_fft_mask_raw)
    max_coord = np.unravel_index(max_pos, img.shape)
    peek_2 = 112 - max_coord[1]

    peek = int((peek_1 + peek_2) / 2)
    close_peek = get_most_close_int(peek)
    moire_peek = 112 - peek

    mask = np.zeros((224, 224, 3), np.uint8)
    cv2.circle(mask, (112 + peek, 112), 5, (1, 1, 1), -1)
    cv2.circle(mask, (112 - peek, 112), 5, (1, 1, 1), -1)
    cv2.circle(mask, (112, 112 + peek), 5, (1, 1, 1), -1)
    cv2.circle(mask, (112, 112 - peek), 5, (1, 1, 1), -1)

    cv2.circle(mask, (112 + close_peek, 112), 5, (1, 1, 1), -1)
    cv2.circle(mask, (112 - close_peek, 112), 5, (1, 1, 1), -1)
    cv2.circle(mask, (112, 112 + close_peek), 5, (1, 1, 1), -1)
    cv2.circle(mask, (112, 112 - close_peek), 5, (1, 1, 1), -1)

    cv2.circle(mask, (112 + moire_peek, 112), 5, (1, 1, 1), -1)
    cv2.circle(mask, (112 - moire_peek, 112), 5, (1, 1, 1), -1)
    cv2.circle(mask, (112, 112 + moire_peek), 5, (1, 1, 1), -1)
    cv2.circle(mask, (112, 112 - moire_peek), 5, (1, 1, 1), -1)

    return mask

def get_most_close_int(peek):
    pixels = 224 / peek
    min_abs = 1e10
    target_value = pixels
    for i in range(1, 6):
        value = i * pixels
        abs_value = abs(round(value) - value)
        if abs_value < min_abs:
            min_abs = abs_value
            target_value = value
    target_value = int( 224 / target_value )
    return target_value

def Moire_fag(img, raw_peak):
    img_fft_abs, img_fft_angle = RGB_fft(img)
    mask = get_mask(img_fft_abs, raw_peak)

    mask1 = 1 - mask

    img_fft_abs_mask = img_fft_abs * mask
    img_fft_abs_mask1 = img_fft_abs * mask1

    img_fft_abs_mask1_noise = add_noise(img_fft_abs_mask1, mask1)
    img_fft_abs_new = img_fft_abs_mask + img_fft_abs_mask1_noise
    img_fft = img_fft_abs_new * np.e ** (1j * img_fft_angle)
    img_aug = RGB_ifft(img_fft)

    return img_aug

img_path = 'IMG_20230521_231421_224_672.jpg'
raw_peak = 95
img = cv2.imread(img_path)

show(img)
img_aug = Moire_fag(img, raw_peak)
show(img_aug)


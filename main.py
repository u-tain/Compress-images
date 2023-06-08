from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
from encoder import encoder_pipeline
from decoder import decoder_pipeline

def run_and_save(jpeg_quality,path):
    pixel_number = 512*512
    all_imgs = []
    bpp = {'baboon':[],'lena':[],'peppers':[],
           'jpeg_lena':[],'jpeg_peppers':[],'jpeg_baboon':[],
           }
    
    psnr = {'baboon':[], 'lena':[], 'peppers':[],
           'jpeg_lena':[], 'jpeg_peppers':[], 'jpeg_baboon':[],
           }

    for items in [8,4,2]:
        B=items
        imgs = []
        for item in ['baboon.png', 'lena.png', 'peppers.png']:
            # производим сжатие изображения
            res = encoder_pipeline(os.path.join(path,item), B=B)
            res2 = decoder_pipeline(os.path.join('results', item[:-4], item[:-4]+f'B{B}.txt'),B)

            # считаем метрики
            names = item[:-4]+f'B{B}'+item[-4:]
            img = Image.open(os.path.join(path,item)).convert("RGB")
            img.load()
            bite_size = os.stat(os.path.join('results',item[:-4],names)).st_size/8
            bpp[item[:-4]].append(bite_size/pixel_number)
            psnr[item[:-4]].append(cv2.PSNR(np.array(img),(res2*255).astype(np.uint8)))

            # посчитаем метрику для изображений сжатых с помощью JPEG2000
            for q in jpeg_quality:
                img = Image.open(os.path.join(path,item)).convert("RGB")
                img.load()
                path_jpeg = 'results/'+item[:-4]+'/'+f'{item[:-4]}{q}.jp2'
                img.convert("RGBA").save(path_jpeg, 'JPEG2000', quality_mode='dB', quality_layers=[q])
                
                bite_size = os.stat(path_jpeg).st_size/8
                bpp['jpeg_'+item[:-4]].append(bite_size/pixel_number)
                psnr['jpeg_'+item[:-4]].append(cv2.PSNR(cv2.imread(os.path.join(path,item)),cv2.imread(path_jpeg)))
            imgs.append(np.concatenate((np.array(img)/255, res2), axis=1))
        all_imgs.append(np.concatenate(imgs, axis=0))
    
    #сохраняем графики
    for item in ['baboon', 'lena', 'peppers']:
        plt.clf()
        plt.plot(bpp[f'jpeg_{item}'],psnr[f'jpeg_{item}'],marker='o')
        plt.plot(bpp[item],psnr[item],marker='o')
        plt.legend(['jpeg','our'])
        plt.xlabel('bpp')
        plt.ylabel('psnr')
        plt.title('psnr/bpp')
        plt.savefig('results/'+item+'/'+item)
        plt.clf()

    plt.imshow(np.concatenate(all_imgs, axis=1))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    jpeg_quality = [19,25,40]
    path = 'test_images'
    # закомментировать следующую строчку, если нужно запустить только энкодер или декодер
    run_and_save(jpeg_quality, path)
    
    # чтобы запустить только энкодер для одного изображения, раскомментировать:
    # B=4 # доступные значения 2,4,8
    # item = 'baboon.png' # название желаемого изображения
    # result = encoder_pipeline(os.path.join(path,item), B=B) #результат будет находиться в results\baboon\baboon.txt

    # чтобы запустить только декодер для одного изображения, раскомментировать:
    # B=4 # доступные значения 2,4,8
    # item = 'baboon.png' # название желаемого изображения
    # result = decoder_pipeline(os.path.join('results', item[:-4], item[:-4]+f'B{B}.txt'),B)

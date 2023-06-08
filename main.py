from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from encoder import encoder_pipeline
from decoder import decoder_pipeline
def make_plot(x1,y1,x2,y2,item):
    plt.clf()
    plt.plot(x1,y1,marker='o')
    plt.plot(x2,y2,marker='o')
    plt.legend(['jpeg','our'])
    plt.xlabel('bpp')
    plt.ylabel('psnr')
    plt.title('psnr/bpp')
    plt.savefig('results/'+item+'/'+item)
    plt.clf()


def run_and_save(jpeg_quality,path):
    pixel_number = 512*512
    all_imgs = []
    bpp = {'baboon':[],'lena':[],'peppers':[],
           'jpeg_lena':[],'jpeg_peppers':[],'jpeg_baboon':[],
           }
    
    psnr = {'baboon':[], 'lena':[], 'peppers':[],
           'jpeg_lena':[], 'jpeg_peppers':[], 'jpeg_baboon':[],
           }

    for items in [2,4,8]:
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
            with open(os.path.join('results',item[:-4],names[:-4]+'.txt'), "r") as f:
                comp = f.read()
            f.close()
            
            bite_size = len(comp)/8 # os.stat(os.path.join('results',item[:-4],names)).st_size/8
            bpp[item[:-4]].append(bite_size/pixel_number)
            psnr[item[:-4]].append(cv2.PSNR(np.array(img),(res2*255).astype(np.uint8)))

            # посчитаем метрику для изображений сжатых с помощью JPEG2000
            imgs.append(np.concatenate((np.array(img)/255, res2), axis=1))
        all_imgs.append(np.concatenate(imgs, axis=0))
    
    #сохраняем графики
    for item in ['baboon', 'lena', 'peppers']:
        for q in jpeg_quality:
                img = Image.open(os.path.join(path,item+'.png')).convert("RGB")
                img.load()
                path_jpeg = 'results/'+item+'/'+f'{item}{q}.jp2'
                img.convert("RGBA").save(path_jpeg, 'JPEG2000', quality_mode='dB', quality_layers=[q])
                bite_size = os.stat(path_jpeg).st_size*8
                print('bite_size ', bite_size, ' ', item, ' ', q)
                bpp['jpeg_'+item].append(bite_size/pixel_number)
                psnr['jpeg_'+item].append(cv2.PSNR(cv2.imread(os.path.join(path,item+'.png')),cv2.imread(path_jpeg)))
        make_plot(bpp[f'jpeg_{item}'],psnr[f'jpeg_{item}'],
                  bpp[item],psnr[item],item)
        # print()

    plt.imshow(np.concatenate(all_imgs, axis=1))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    jpeg_quality = [30,45,55]
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

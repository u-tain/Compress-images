from model import Encoder, Decoder
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T
import os
import cv2

def encoder_pipeline(image_path,B=2):
    encoder = Encoder(B)
    encoder.load_state_dict(torch.load(f'weights\encoder_weights_b{B}.pth'))
    encoder.eval()
    image = Image.open(image_path).convert("RGB")
    image.load()
    image = np.array(image)
    # plt.imshow(image)
    # plt.show()
    image = image/255
    image = T.ToTensor()(image).float()
    image = image.reshape(1,3,512,512)
    res = encoder(image) # квантование входит в модель
    # арифметическое кодирование 
    return res
    
def decoder_pipeline(compress_image, B):
    decoder = Decoder(B).eval()
    decoder.load_state_dict(torch.load(f'weights\decoder_weights_b{B}.pth'))
    res = decoder(compress_image.float())
    res = torch.sigmoid(res).detach().cpu().numpy()
    # plt.imshow(res[0].squeeze().transpose(1,2,0))
    # plt.show()
    return res

if __name__ == '__main__':
    pixel_number = 512*512
    jpeg_quality = [19,13,40]
    path = 'test_images'
    all_imgs = []
    bpp = {'baboon':[],'lena':[],'peppers':[],
           'jpeg_lena':[],'jpeg_peppers':[],'jpeg_baboon':[],
           }
    
    psnr = {'baboon':[], 'lena':[], 'peppers':[],
           'jpeg_lena':[], 'jpeg_peppers':[], 'jpeg_baboon':[],
           }
    
    for items in [8]:
        B=items
        imgs = []
        for item in ['baboon.png', 'lena.png', 'peppers.png']:
            # производим сжатие изображения
            res = encoder_pipeline(os.path.join(path,item), B=B)
            res2 = decoder_pipeline(res,B)
            res2 = res2[0].squeeze().transpose(1,2,0)

            # сохраняем результат сжатия
            file = Image.fromarray((res2*255).astype(np.uint8))
            names = item[:-4]+f'B{B}'+item[-4:]
            file.save(os.path.join('results',item[:-4],names))

            # считаем метрики
            img = Image.open(os.path.join(path,item)).convert("RGB")
            img.load()
            bite_size = os.stat(os.path.join('results',item[:-4],names)).st_size/8
            bpp[item[:-4]].append(bite_size/pixel_number)
            psnr[item[:-4]].append(cv2.PSNR(np.array(img),(res2*255).astype(np.uint8)))

            # посчитаем метрику для изоброажений сжатых с помощью JPEG2000
            for q in jpeg_quality:
                path_jpeg = 'results/'+item[:-4]+'/'+f'{item[:-4]}{q}.jp2'
                img.convert("RGBA").save(path_jpeg, 'JPEG2000', quality_mode='dB', quality_layers=[q])
                
                bite_size = os.stat(path_jpeg).st_size/8
                bpp['jpeg_'+item[:-4]].append(bite_size/pixel_number)
                psnr['jpeg_'+item[:-4]].append(cv2.PSNR(cv2.imread(os.path.join(path,item)),cv2.imread(path_jpeg)))

            #сохраняем графики
            plt.clf()
            plt.plot(bpp[f'jpeg_{item[:-4]}'],psnr[f'jpeg_{item[:-4]}'],marker='o')
            plt.plot(bpp[item[:-4]],psnr[item[:-4]],marker='o')
            plt.legend(['jpeg','our'])
            plt.xlabel('bpp')
            plt.ylabel('psnr')
            plt.title('psnr/bpp')
            plt.savefig('results/'+item[:-4]+'/'+item)
            # imgs.append(np.concatenate((np.array(img)/255, res2), axis=1))
        # all_imgs.append(np.concatenate(imgs, axis=0))
    # plt.imshow(np.concatenate(all_imgs, axis=1))
    # plt.show()

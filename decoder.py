import numpy as np
from haffman import  decode_haff
from model import  Decoder
import torch
from PIL import Image
import os

def decoder_pipeline(path_to_compress_image, B):
    # восстановление с помощью кода хаффмана
    with open(path_to_compress_image, "r") as f:
        compress_image = f.read()
    f.close()
    compress_image = decode_haff(compress_image)
    compress_image = np.fromstring(compress_image, dtype=int, sep=' ')
    compress_image = torch.from_numpy(compress_image).reshape(1,compress_image.shape[0])
    
    # декодирование сжатого изображения
    decoder = Decoder(B).eval()
    decoder.load_state_dict(torch.load(f'weights\decoder_weights_b{B}.pth'))
    res = decoder(compress_image.float())
    res = torch.sigmoid(res).detach().cpu().numpy()
    res = res[0].squeeze().transpose(1,2,0)

    # сохраняем результат сжатия
    img_name = path_to_compress_image[:-4] 
    file = Image.fromarray((res*255).astype(np.uint8))
    file.save(img_name+'.png')
    return res
import torch
import numpy as np
from model import Encoder
from haffman import encode_haff
from PIL import Image
from torchvision import transforms as T


def encoder_pipeline(image_path,B=2):
    encoder = Encoder(B)
    encoder.load_state_dict(torch.load(f'weights\encoder_weights_b{B}.pth'))
    encoder.eval()
    image = Image.open(image_path).convert("RGB")
    image.load()
    image = np.array(image)
    image = image/255
    image = T.ToTensor()(image).float()
    image = image.reshape(1,3,512,512)
    res = encoder(image) # квантование входит в модель
    res = torch.floor(res)
    # кодирование хаффмана
    str_from_arr = ' '.join([str(int(item)) for item in res[0].detach().cpu().numpy()])
    res = encode_haff(str_from_arr,B)
    name = 'results/'+image_path[12:-4]+'/'+image_path[12:-4]+f'B{B}'+".txt"
    with open(name, "w") as file:
        file.write(res)
    file.close()
    return res
    
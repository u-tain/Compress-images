import json
from model import Encoder
from PIL import Image
import torch 
import numpy as np
from torchvision import transforms as T
import os


class Node(object):
    def __init__(self, name=None, value=None):
        self.name = name
        self.value = value
        self.lchild = None
        self.rchild = None

# дерево Хаффмана
class HuffmanTree(object):
    def __init__(self, char_Weights):
        self.Leaf = [Node(k,v) for k, v in char_Weights.items()]
        while len(self.Leaf) != 1:
            self.Leaf.sort(key=lambda node:node.value, reverse=True)
            n = Node(value=(self.Leaf[-1].value + self.Leaf[-2].value))
            n.lchild = self.Leaf.pop(-1)
            n.rchild = self.Leaf.pop(-1)
            self.Leaf.append(n)
        self.root = self.Leaf[0]
        self.Buffer = list(range(10))
        self.dict_for_json = {}

    def Hu_generate(self, tree, length):
        node = tree
        if (not node):
            return
        elif node.name:
            self.dict_for_json[node.name] = ''
            for i in range(length):
                self.dict_for_json[node.name]+=str(self.Buffer[i])
            return
        self.Buffer[length] = 0
        self.Hu_generate(node.lchild, length + 1)
        self.Buffer[length] = 1
        self.Hu_generate(node.rchild, length + 1)

    #Output кодировка Хаффмана
    def get_code(self):
        self.Hu_generate(self.root, 0)

def build_tree(image_path,B=2):
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
    return res

def encode_haff(s):
    with open('to_huff.json') as json_file:
        data = json.load(json_file)
    res = ''
    for item in s:
        res+=data[item]
    return res

def decode_haff(s):
    with open('from_huff.json') as json_file:
        data = json.load(json_file)
    keys = list(data.keys())
    res = ''
    key = ''
    for item in s:
        key += item
        if key in keys:
            res += data[key]
            key=''
    return res

if __name__=='__main__':
    path = 'test_images'
    keys = ['0','1','2','3','4','5','6','7','8','9',' ']
    haff_dict = dict.fromkeys(keys,0)
    lens = 0
    for items in [4,8]:
        B=items
        for item in ['baboon.png', 'lena.png', 'peppers.png']:
            res = build_tree(os.path.join(path,item), B=B)
            str_from_arr = ' '.join([str(int(el)) for el in res[0].detach().cpu().numpy()])
            lens += len(str_from_arr)
            for k in keys:
                haff_dict[k]+= str_from_arr.count(k)
    for k in keys:
        haff_dict[k] = round(haff_dict[k]/lens,3)
        
    tree = HuffmanTree(haff_dict)
    tree.get_code()
    code = tree.dict_for_json

    # сохраняем словари чтобы было удобно кодировать и декодировать
    jsonString = json.dumps(code)
    jsonFile = open("to_huff.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    jsonString = json.dumps(dict(zip(list(code.values()),list(code.keys()))))
    jsonFile = open("from_huff.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

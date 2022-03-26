import argparse
import os
import cv2

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from deepface.backbones.iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from deepface.backbones.mobilefacenet import get_mbf
from deepface.commons import functions
import gdown
url={
    'ms1mv3_r50':'https://eb9uqq.dm.files.1drv.com/y4mo1LyxVkMS7RwyNFyD7Oj_LrukPmnMwHsL9rjh0By0Pbgglx-f55KwzpQ7rMhHYsgqz8WXcFOFpNKcgwBwPpmd2UjEOc2JwcdRAitVfngManBko6wU-y2HTwGi--_4R9_TmfTqO4yGQEIhR9-d4LOcisKC8YzL4bth1b4tSJ8nloIIq7xGizPX3jWiYfFHzirG5-VgJ3guFBVZKE7pupRsw',
    'ms1mv3_r18':'https://eb9uqq.dm.files.1drv.com/y4mpJ0NiyBPDzo_aQlh9QHwL52UljHSI60KSPv0-p2oTb4qnoUA5Cu3Ul-Tfxc8l7uyg9BYE_hoItNc9JjqYRW-qmIIM0JeMqKGjyl5sZQvwPZUxazPW8THT9CrWpwzaKkrBXFDc_uEDGAvDpaB1lhrc83aG5lBOeuI6LbtMLBHyR7TA2YdPxcIvPGnsbqjvWl1rXQFG4zD2_TxL_m4avN43Q',
    'ms1mv3_r34': 'https://eb9uqq.dm.files.1drv.com/y4mU3JhshWSlooEzKRYnCPrOb1-xpZqS_Z90rOXm8D6KOL-PpOhvlsDYAgiTWkGG8TYqC2kdgr4I66XBkhEtqhptKTRFY90gnLTesR9Sw0xNGb46_ULn6IcfRMTW18uKJS2pwGpwabu7SpL3Z1EsX-gcd74M26gMJ11svjthg15CzpGQhVASMZMMfSvlUGhyP5HPFxOQi3X0cpAUMm8P9Yn8Q',
    'ms1mv3_r100':'https://eb9uqq.dm.files.1drv.com/y4mNdH0KjE7_R3tIT1h86Ov1XshRRgT1BUBeVIrUgRasS5x93UeCpP023bspth03rUtIg1raK3EtRqMtrGf_DvA0pIf2RgB7FsHsBaNoJYF1JqUl7Q8qsTpYGxOaq7-ow0Hiejjz5JRU9nWOJSniOlM2STvDKZH-Zs6pHiyLEfLhikQkm8xC2SYkcas-xedihqRJCVmzTI4LfBqtFbX1nxU-Q',
    'glint360_r18':'https://eb9uqq.dm.files.1drv.com/y4mn1hArpddPJw-OM6IzTll6TpxZaSVjs6HyzeYC2m-tg-v9qqBjoI37Lr20K-RNFr-9_AlbnguKxxzrC4lqSykaUNWaJhya12ZdOIIwS1h2kPGSjGJkCEyEca9YkV5Mkesiee8nHibkeLvY5uSoe5PSLtm_umgqd6l3f4-RSnP4ecGrtYM3-Jt49YgKPwDcb5hNyXVBixUqVhTmyOiw9pM3g',
    'glint360_r34': 'https://eb9uqq.dm.files.1drv.com/y4mDEvblVeT7MFazuuzSCum_bzCqIrvBi8aubLVv33z28NOLEBsT9tPGX4FGk1vpUUz-9evFi94M_B5rPkJs6mLovRBA32EQZWQv8q5wSnoapg7suxh3RVxH-veJqTHuolx4d-OvvbYlMfVCts-HwU2jFP1gFfhyvh2q0CDLcAudda_TrxtxgBQGZhv1sun9twAJDycUVD5cmKumsxhlqvh3A',
    'glint360_r50': 'https://eb9uqq.dm.files.1drv.com/y4m7HMGc6qBhL2PwUcsjx4z-Pm57HD2Uze1oa27yGL4BXt4Ech3sIbi59XUpBJMv6kxAAxJP00W_lWyN8T8Dm2rZ8eLQVxMiNoskpN0JZOfjTeiovnhNwBsOc3RN2Y91xNqzyMPs-5GQ4qKdZ_LNlulu8wckJcWvTIFSupsLkmtnym8PnL5u7XTERhXBTgL5nwoutQg6Yvb8Ixr_5VY1m2LaQ',
    'glint360_r100': 'https://eb9uqq.dm.files.1drv.com/y4m6MECUN2ituEEi6oi8ksrTVHaNKfu21zaqpVA750ynYQqsP-RSDbGFX_MyK-OdWOnFp9NZuFTU711TVGAUMbttVWclSzruJRQUEp7-D8fZLMUBPc43lXSAkReo6WCfWaHIFZltEsfO3WomoCyePTRlEgShXYxVpSnu_VDuD8_MC7WcRmBJGznahexUgSQE0NcVJDvYkq2MW1eaeEQ0T4d6Q'
    }
def getmodel(name, **kwargs):
    # resnet
    if name == "r18":
        base_model= iresnet18(False, **kwargs)
    elif name == "r34":
        base_model= iresnet34(False, **kwargs)
    elif name == "r50":
        base_model= iresnet50(False, **kwargs)
    elif name == "r100":
        base_model= iresnet100(False, **kwargs)
    elif name == "r200":
        base_model= iresnet200(False, **kwargs)
    elif name == "r2060":
        from deepface.backbones.iresnet2060 import iresnet2060
        base_model= iresnet2060(False, **kwargs)
    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        base_model= get_mbf(fp16=fp16, num_features=num_features)
    else:
        raise ValueError()
    return base_model
class Model_ArcFace(nn.Module):
    def __init__(self,name,weight):
        super().__init__()
        self.model= getmodel(name, fp16=False)
        self.model.load_state_dict(torch.load(weight, map_location=torch.device("cpu") ))
        self.model.eval()
    @torch.no_grad()
    def predict(self,image):
        self.img=image
        # self.img = np.transpose(  self.img, (0,3,1,2))
        # self.img = torch.from_numpy(self.img).float()
        self.img = np.transpose(self.img, (0,3, 1, 2))
        self.img = torch.from_numpy(self.img).float()
        # self.img.div_(255).sub_(0.5).div_(0.5)
        # print(self.img.shape)
        feat = self.model(self.img)
        feat=feat.numpy()
        return feat
  
def loadModel_ms1mv3_r50(url = 'https://eb9uqq.dm.files.1drv.com/y4mo1LyxVkMS7RwyNFyD7Oj_LrukPmnMwHsL9rjh0By0Pbgglx-f55KwzpQ7rMhHYsgqz8WXcFOFpNKcgwBwPpmd2UjEOc2JwcdRAitVfngManBko6wU-y2HTwGi--_4R9_TmfTqO4yGQEIhR9-d4LOcisKC8YzL4bth1b4tSJ8nloIIq7xGizPX3jWiYfFHzirG5-VgJ3guFBVZKE7pupRsw'):
    home = functions.get_deepface_home()
    file_name = "backbone.pth"
    output = home+'/.deepface/weights/ms1mv3_arcface_r50/'+file_name
    if os.path.exists(output) != True and  os.path.exists(home+'/.deepface/weights/ms1mv3_arcface_r50/') !=True :
        os.mkdir(home+'/.deepface/weights/ms1mv3_arcface_r50/')
        print(file_name," will be downloaded to ",output)
        gdown.download(url, output, quiet=False)

    model=Model_ArcFace('r50',output)
    return model
def loadModel(name):
    home = functions.get_deepface_home()
    file_name = "backbone.pth"
    output= home + '/.deepface/weights/'+name+"/"+file_name
    if os.path.exists(output) != True:
        os.mkdir(home+ '/.deepface/weights/'+name+"/")
        print(file_name," will be downloaded to ",output)
        gdown.download(url[name], output, quiet=False)     
    name_model=name.split("_")[-1]  
    model= Model_ArcFace(name_model,output)
    return model
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
#     parser.add_argument('--model_name', type=str, default='glint360_r100', help='backbone network')
#     parser.add_argument('--img', type=str, default='/home/quang/Documents/FACE/deepface/tests/dataset/img1.jpg')
#     args = parser.parse_args()
#     model_name=args.model_name
#     path_img=args.img
#     model=loadModel(model_name)
#     first_parameter = next(model.parameters())
#     input_shape = first_parameter.size()
#     input_shape=(112,112)
#     # input_shape = model.layers[0].input_shape
#     print(input_shape)
#     img1 = functions.preprocess_face(path_img,input_shape)
#     feat=model.predict(img1)
#     print(feat.shape)

#     img1 = functions.preprocess_face("tests/dataset/img1.jpg", input_shape)
        
#     img1_representation = model.predict(img1)[0,:]
#     #img2 = functions.detectFace("dataset/img3.jpg", input_shape)
#     img2 = functions.preprocess_face("tests/dataset/img3.jpg", input_shape)
#     img2_representation = model.predict(img2)[0,:]


#     print("img2_representation",img2_representation.shape)
#     #----------------------------------------------
#     #distance between two images

#     distance_vector = np.square(img1_representation - img2_representation)
#     #print(distance_vector)

#     distance = np.sqrt(distance_vector.sum())
#     print("Euclidean distance: ",distance)

#     #----------------------------------------------
#     #expand vectors to be shown better in graph

#     img1_graph = []; img2_graph = []; distance_graph = []
#     for i in range(0, 200):
#         img1_graph.append(img1_representation)
#         img2_graph.append(img2_representation)
#         distance_graph.append(distance_vector)

#     img1_graph = np.array(img1_graph)
#     img2_graph = np.array(img2_graph)
#     distance_graph = np.array(distance_graph)

#     #----------------------------------------------
#     #plotting

#     fig = plt.figure()

#     ax1 = fig.add_subplot(3,2,1)
#     plt.imshow(img1[0][:,:,::-1])
#     plt.axis('off')

#     ax2 = fig.add_subplot(3,2,2)
#     im = plt.imshow(img1_graph, interpolation='nearest', cmap=plt.cm.ocean)
#     plt.colorbar()

#     ax3 = fig.add_subplot(3,2,3)
#     plt.imshow(img2[0][:,:,::-1])
#     plt.axis('off')

#     ax4 = fig.add_subplot(3,2,4)
#     im = plt.imshow(img2_graph, interpolation='nearest', cmap=plt.cm.ocean)
#     plt.colorbar()

#     ax5 = fig.add_subplot(3,2,5)
#     plt.text(0.35, 0, "Distance: %s" % (distance))
#     plt.axis('off')
#     ax6 = fig.add_subplot(3,2,6)
#     im = plt.imshow(distance_graph, interpolation='nearest', cmap=plt.cm.ocean)
#     plt.colorbar()


#     plt.show()

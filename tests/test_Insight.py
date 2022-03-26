
import argparse
from deepface.basemodels.InsightFace import loadModel_r50
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    weight,name,img=args.weight, args.network, args.img
    
    
    model=loadModel_r50()

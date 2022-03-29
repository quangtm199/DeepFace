#!pip install deepface
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace , InsightFace , ArcFace
from deepface.commons import functions 

import matplotlib.pyplot as plt
import numpy as np

#----------------------------------------------
#build face recognition model

# model = VGGFace.loadModel()
model = InsightFace.loadModel('ms1mv3_r100')
# model = Facenet.loadModel()
#model = OpenFace.loadModel()
#model = FbDeepFace.loadModel()
try:
	input_shape = model.layers[1].input_shape[1:3]
	print("model input shape: ", model.layers[0].input_shape[1:])
	print("model output shape: ", model.layers[-1].input_shape[-1])

except:
	input_shape=(112,112)
# input_shape=(112,112)
import cv2
import torch
#----------------------------------------------
#load images and find embeddings
# img1 = cv2.imread("/home/quang/Documents/FACE/deepface/1.jpg")
# img1 = cv2.resize(img1, (112, 112))
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# img1 = np.transpose(img1, (2, 0, 1))
# img1 = torch.from_numpy(img1).unsqueeze(0).float()
# img1.div_(255).sub_(0.5).div_(0.5)
# img = cv2.imread("/home/quang/Documents/FACE/deepface/2.jpg")
# img = cv2.resize(img, (112, 112))

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = np.transpose(img, (2, 0, 1))
# img2 = torch.from_numpy(img).unsqueeze(0).float()
# img2.div_(255).sub_(0.5).div_(0.5)
# img1 = functions.detectFace("dataset/img1.jpg", input_shape)
img2 = functions.preprocess_face("/home/quang/Documents/FACE/deepface/tests/dataset/img36.jpg", input_shape)
img1 = functions.preprocess_face("/home/quang/Documents/FACE/deepface/tests/dataset/img61.jpg", input_shape)
# img11="/home/quang/Documents/FACE/deepface/1.jpg"
# print("img1",img1.shape)
# img1_representation = model.predict1(img11)[0,:]
img1_representation = model.predict(img1)[0,:]
# img22="/home/quang/Documents/FACE/deepface/2.jpg"
# img2 = functions.detectFace("dataset/img3.jpg", input_shape)
img2_representation = model.predict(img2)[0,:]
# img2_representation = model.predict1(img22)[0,:]
# print(img2_representation)

print("img2_representation",img2_representation.shape)
#----------------------------------------------
#distance between two images

distance_vector = np.square(img1_representation - img2_representation)
#print(distance_vector)
distance = np.sqrt(distance_vector.sum())
print("Euclidean distance: ",distance)
#----------------------------------------------
#expand vectors to be shown better in graph

img1_graph = []; img2_graph = []; distance_graph = []
for i in range(0, 200):
	img1_graph.append(img1_representation)
	img2_graph.append(img2_representation)
	distance_graph.append(distance_vector)

img1_graph = np.array(img1_graph)
img2_graph = np.array(img2_graph)
distance_graph = np.array(distance_graph)
#----------------------------------------------
#plotting

fig = plt.figure()

ax1 = fig.add_subplot(3,2,1)
plt.imshow(img1[0][:,:,::-1])
plt.axis('off')

ax2 = fig.add_subplot(3,2,2)
im = plt.imshow(img1_graph, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()

ax3 = fig.add_subplot(3,2,3)
plt.imshow(img2[0][:,:,::-1])
plt.axis('off')

ax4 = fig.add_subplot(3,2,4)
im = plt.imshow(img2_graph, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()

ax5 = fig.add_subplot(3,2,5)
plt.text(0.35, 0, "Distance: %s" % (distance))
plt.axis('off')

ax6 = fig.add_subplot(3,2,6)
im = plt.imshow(distance_graph, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()

plt.show()

#----------------------------------------------
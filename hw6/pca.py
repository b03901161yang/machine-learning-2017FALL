import numpy as np
from skimage import io
from numpy import linalg as LA
import copy
import sys
import os

file_path = sys.argv[1]
image_name = sys.argv[2]

target_image_name = os.path.join(sys.argv[1],sys.argv[2])

your_list = os.listdir(file_path)
#print('len your list',len(your_list))
#print('your list', your_list[0])
#print('target_image_name', target_image_name)

flat_img_arr = np.zeros((600*600*3, 415))
for count in range(415): #415 images
    file_name = os.path.join( file_path , your_list[count])
    img = io.imread(file_name)
    img = img/255.0
    flat_img = img.flatten()
    flat_img_arr[:,count] = flat_img


mean_img = np.mean(flat_img_arr, axis=1)
#print('mean img shape: ', mean_img.shape ) #600*600*3


#mean_img_2d = np.reshape(mean_img,(600,600,3))
#mean_img_2d = (mean_img_2d*255).astype(np.uint8)

#io.imsave('meanImage.jpg', mean_img_2d)
#io.imshow(mean_img_2d)
#io.show()


for count in range(415):
    flat_img_arr[:,count] -= mean_img

#print('flat_img_arr shape: ', flat_img_arr.shape )

#img_gram_matrix = np.dot( np.transpose(flat_img_arr), flat_img_arr)
#print('img_gram_matrix shape: ', img_gram_matrix.shape )

xTv,s,V = np.linalg.svd(flat_img_arr, full_matrices= 0) #100 eigen faces

#eig_face = np.dot(flat_img_arr, xTv)
eig_face = xTv


#print('xTv shape: ', xTv.shape )
#print('eig_face shape: ', eig_face.shape )
#print('s shape: ', s.shape )
#print('s 0:10: ', s[0:10,] )

num_of_eig = 4
eig_face = eig_face[:,0:num_of_eig]

'''
for n in range(num_of_eig):
    eig_face[:,n] /= LA.norm(eig_face[:,n])
'''

#print('eig face 1:',eig_face)

'''
for n in range(num_of_eig):
    eig_face_copy = copy.copy(eig_face)
    tmp_eig_face = np.reshape(eig_face_copy[:,n],(600,600,3))
    tmp_eig_face -= np.min(tmp_eig_face)
    tmp_eig_face /= np.max(tmp_eig_face)
    tmp_eig_face = (tmp_eig_face*255).astype(np.uint8)
    #io.imsave('eigen'+str(n)+'.jpg', tmp_eig_face)
'''

'''
ratio_arr = np.zeros((4,))

sum_singular_value = np.sum(s)

print('sum_singular_value shape: ', sum_singular_value.shape )

for n in range(4):
    ratio_arr[n,] = s[n,]/sum_singular_value
    print('largest ',n,' eigenvector ratio: ', ratio_arr[n,])


print('eig_face shape: ', eig_face.shape )
'''


target_image = io.imread(target_image_name)
img = target_image
img = img / 255.0
img = img.flatten()
img -= mean_img


reconstruct_coef = np.dot( img, eig_face)
print('reconstruct_coef ',reconstruct_coef)
#print('eig_face ', eig_face.shape)
reconstruct_image_tmp = np.dot( eig_face, reconstruct_coef) + mean_img
reconstruct_image_tmp = np.reshape(reconstruct_image_tmp,(600,600,3))
reconstruct_image_tmp -= np.min(reconstruct_image_tmp)
reconstruct_image_tmp /= np.max(reconstruct_image_tmp)
reconstruct_image_tmp = (reconstruct_image_tmp*255).astype(np.uint8)
io.imsave('reconstruction.jpg',reconstruct_image_tmp)

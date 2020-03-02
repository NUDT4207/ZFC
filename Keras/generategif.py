# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:33:15 2019

@author: x1c
"""

import imageio 



def gif(name,ite,file_name):
    filenames = []
    for i in range(ite):
        if i%2==0:
            filename = file_name+'xf'+str(i)+'.png'
            print(filename)
            filenames.append(filename)
    
    img_paths = filenames 
    gif_images = [] 
    nums=1
    for path in img_paths: 
        gif_images.append(imageio.imread(path)) 
        print(nums)
        nums=nums+1
    imageio.mimsave('./gif_low/'+str(name)+'.gif',gif_images,fps=20)


if __name__=='__main__':
    name_space = ['w_0.1_b_1.0','w_0.1_b_0.1','w_0.01_b_1e-05',
                  'w_0.01_b_0.0001','w_0.01_b_0.001','w_0.01_b_0.01',
                  'w_0.01_b_1.0','w_0.01_b_0.1','w_0.1_b_1e-05',
                  'w_0.1_b_0.001','w_0.1_b_0.0001','w_0.1_b_0.01']
    name_space = ['w_0.1_b_0.01']
    for name in name_space:
        file_name = './pics/'+name+'/'
        gif(name,ite=200,file_name=file_name)
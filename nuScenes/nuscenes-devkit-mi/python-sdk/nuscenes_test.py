#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""/********************************************************************/
    > File Name: nuscenes_test.py
    > Author: hegaozhi
    > Mail: hzg@szlanyou.com 
    > Created Time: 2021年02月25日 星期四 10时32分03秒
    > https://www.nuscenes.org/nuscenes?tutorial=nuscenes
/********************************************************************/"""


import sys
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import numpy as np
from PIL import Image

# 初始化
nusc = NuScenes(version='v1.0-mini', dataroot='/home/hegaozhi/hegaozhi/tools/calibrate/data/nuScenes/v1.0-mini', verbose=True)

# 列出所有的场景
print('all scene')
nusc.list_scenes() 

# 列出所有场景中的第一个场景对应的所有 token     
my_scene = nusc.scene[0]
print('\nmy_scene\n', my_scene)         

 # scene.json 中第一个元素组中的 first_sample_token last_sample_token
first_sample_token = my_scene['first_sample_token']     
# sample.json 中第一帧的所有元素(token、timestamp、prev、next、scene_token) 和汽车所有传感器对应的数据
my_sample = nusc.get('sample', first_sample_token)
print('my_sample\n', my_sample)
#       只取汽车所有传感器对应的数据的方法 print('my_sample\n', my_sample['data'])
#       只取CAM_FRONT 的方法
#sensor = 'CAM_FRONT'
#cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
#print(cam_front_data)
#               以此类推
#       获取传感器对应的标注框的方法(以前一个例子为例)[可视化]
#nusc.render_sample_data(cam_front_data['token'])

nusc.list_sample(my_sample['token'])      # 列出了与示例相关的所有sample_data关键帧和sample_annotation

my_annotation_token = my_sample['anns'][18]  # 取第19个 sample_annotation_token（其它的好像没标注好）
my_annotation_metadata = nusc.get('sample_annotation', my_annotation_token)     # 得到第一个的详细注释 sample_annotation.json 的一个元素组
print('\nmy_annotation_metadata\n', my_annotation_metadata)
#nusc.render_annotation(my_annotation_token)     # [可视化]
my_instance = nusc.instance[599]        # instance.json 的 4196 行开始 ？
print('my_instance\n', my_instance)


# 转换点云坐标，使其与像素坐标系一致
nusc_exp = NuScenesExplorer(nusc)
#points, coloring, image = nusc_exp.map_pointcloud_to_image(my_sample['data']['RADAR_FRONT_LEFT'],my_sample['data']['CAM_FRONT_LEFT'])
nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel = 'RADAR_FRONT', camera_channel = 'CAM_FRONT', dot_size=50)


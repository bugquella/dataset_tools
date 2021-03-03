import numpy as np
import mayavi.mlab
import math
import sys
import os

from numpy import cos,sin,pi,arange
from traits.api import HasTraits,Instance,Range,on_trait_change
from traitsui.api import View,Item,Group
from mayavi.core.ui.api import MayaviScene,SceneEditor,MlabSceneModel
from mayavi.core.api import PipelineBase


fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))  
     
def lidar_show(pc0):
    x = pc0[:, 0]  # x position of point
    y = pc0[:, 1]  # y position of point
    z = pc0[:, 2]  # z position of point
    r = pc0[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
    degr = np.degrees(np.arctan(z / d))
    col = z
                                               
    mayavi.mlab.points3d(x, y, z,
                         col,  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )
    
    
    # 绘制原点
    mayavi.mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere",scale_factor=1)
    # 绘制坐标
    axes = np.array(
        [[20.0, 0.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0], [0.0, 0.0, 20.0, 0.0]],
        dtype=np.float64,
    )
    #x轴
    mayavi.mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig,
    )
    #y轴
    mayavi.mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig,
    )
    #z轴
    mayavi.mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig,
    )
    #mayavi.mlab.show()
    print(pc0.shape)
    
if __name__ == '__main__':
    pointcloud = np.fromfile(str("/home/l3plus/hegaozhi/data/pandaset/pandaset/pandaset-devkit-master/000000.bin"), dtype=np.float32, count=-1).reshape([-1, 4])
    lidar_show(pointcloud)
    myName = input()
    pointcloud = np.fromfile(str("/home/l3plus/hegaozhi/data/pandaset/pandaset/pandaset-devkit-master/000001.bin"), dtype=np.float32, count=-1).reshape([-1, 4])
    print('--')
    lidar_show(pointcloud)
    myName = input()
    
    

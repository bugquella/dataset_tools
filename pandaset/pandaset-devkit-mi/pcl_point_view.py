import numpy as np
import pcl.pcl_visualization
import time
# lidar_path 指定一个kitti 数据的点云bin文件就行了
points = np.fromfile("/home/l3plus/hegaozhi/data/pandaset/pandaset/pandaset-devkit-master/000000.bin", dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)


# 这里对第四列进行赋值，它代表颜色值，根据你自己的需要赋值即可；
#points[:, 3] =  points[:, 2]
# PointCloud_PointXYZRGB 需要点云数据是N*4，分别表示x,y,z,RGB ,其中RGB 用一个整数表示颜色；
color_cloud = pcl.PointCloud_PointXYZI(points)

visual = pcl.pcl_visualization.CloudViewing()
visual.ShowGrayCloud(color_cloud, b'cloud')
print('+++')
time.sleep(100)
visual.ShowGrayCloud(color_cloud, b'cloud')
print('+++')
time.sleep(100)
print('+++')
visual.ShowGrayCloud(color_cloud, b'cloud')
print('+++')
time.sleep(100)

#flag = True
#while flag:
#    flag != visual.WasStopped()

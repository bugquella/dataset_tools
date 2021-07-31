#coding:utf-8

from __future__ import print_function

import os
import sys
import numpy as np
import cv2

import numpy as np
import cv2
import os, math


# 存放矫正参数的类
class Calib:
    def __init__(self, dict_calib):
        super(Calib, self).__init__()
        self.P0 = dict_calib['P0'].reshape((3, 4))
        self.P1 = dict_calib['P1'].reshape((3, 4))
        self.P2 = dict_calib['P2'].reshape((3, 4))
        self.P3 = dict_calib['P3'].reshape((3, 4))
        self.R0_rect = dict_calib['R0_rect'].reshape((3, 3))
        self.Tr_velo_to_cam = dict_calib['Tr_velo_to_cam'].reshape((3, 4))
        self.Tr_imu_to_velo = dict_calib['Tr_imu_to_velo'].reshape((3, 4))
        
    # 将点云数据变成齐次矩阵形式
    def cart2homo(self, pcs_vel):
        # 输入： 3*N
        # 输出： 4*N
        return np.vstack((pcs_vel, np.ones((1, pcs_vel.shape[1]))))
 
    # 将世界坐标系上的点云投影到相机坐标系下的
    def project_vel_to_cam(self, pcs_vel):
        # 输入：3*N
        # 输出：4*N
        # 先将点云数据（3*N）变成其次矩阵形式
        pcs_homo = self.cart2homo(pcs_vel)
        pcs_cam = np.dot(self.Tr_velo_to_cam, pcs_homo)
        return pcs_cam

    # 将点云进行修正
    def rectify_pcs(self, pcs_in_cam):
        # 输入：3*N
        # 输出：3*N
        return np.dot(self.R0_rect, pcs_in_cam)
    
    # 利用内参矩阵将修正后的点云投影到像素坐标系下
    def project_rected_to_image(self, pcs_rected):
        pcs_2d = np.dot(self.P2, np.vstack((pcs_rected, np.ones((1, pcs_rected.shape[1])))))
        pcs_2d[0, :] /= pcs_2d[2, :]
        pcs_2d[1, :] /= pcs_2d[2, :]
        return pcs_2d[:-1, :]

    # 将点云图转成2d图像
    def project_vel_to_image(self, pcs_3d):
        # 输入点云数据：3*N
        # 输出：2*N
        # 利用外参矩阵将在世界坐标系下的点云数据转到相机坐标系下
        pcs_in_cam = self.project_vel_to_cam(pcs_3d)
        pcs_rected = self.rectify_pcs(pcs_in_cam)
        pcs_2d = self.project_rected_to_image(pcs_rected)
        return pcs_2d


# 存放标签的类
class Object3d:
    def __init__(self, content):
        super(Object3d, self).__init__()
        # content就是一个字符串，根据空格分割开来
        lines = content.split()
        # 去掉空字符
        lines = list(filter(lambda x: len(x), lines))
        self.name, self.truncated, self.occluded, self.alpha = lines[0], float(lines[1]), float(lines[2]), float(
            lines[3])
        self.bbox = [lines[4], lines[5], lines[6], lines[7]]
        self.bbox = np.array([float(x) for x in self.bbox])
        self.dimensions = [lines[8], lines[9], lines[10]]
        self.dimensions = np.array([float(x) for x in self.dimensions])
        self.location = [lines[11], lines[12], lines[13]]
        self.location = np.array([float(x) for x in self.location])
        self.rotation_y = float(lines[14])


# KITTI数据集的类
class Kitti_Dataset:
    def __init__(self, dir_path, split="training"):
        super(Kitti_Dataset, self).__init__()
        self.dir_path = os.path.join(dir_path, split)
        # calib矫正参数文件夹地址
        self.calib = os.path.join(self.dir_path, "calib")
        # RGB图像的文件夹地址
        self.images = os.path.join(self.dir_path, "image_2")
        # 点云图像的文件夹地址
        self.pcs = os.path.join(self.dir_path, "velodyne")
        # 标签文件夹的地址
        self.labels = os.path.join(self.dir_path, "label_2")
 
    # 得到当前数据集的大小
    def __len__(self):
        file = []
        for _, _, file in os.walk(self.images):
            pass
        # 返回rgb图片的数量
        return len(file)
 
    # 得到矫正参数的信息
    def get_calib(self, index):
        # 得到矫正参数文件
        calib_path = os.path.join(self.calib, "{:06d}.txt".format(index))
        # 读取文件，每一个都表示一个参数矩阵，我们用Calib这个类来表示
        with open(calib_path) as f:
            lines = f.readlines()
        # 去掉空行和换行符
        lines = list(filter(lambda x: len(x) and x != '\n', lines))
        dict_calib = {}
        for line in lines:
            key, value = line.split(':')
            dict_calib[key] = np.array([float(x) for x in value.split()])
        return Calib(dict_calib)
 
    # 获取rgb图片的方法
    def get_rgb(self, index):
        # 首先得到图片的地址
        img_path = os.path.join(self.images, "{:06d}.png".format(index))
        return cv2.imread(img_path)
 
    # 获取点云数据的方法
    def get_pcs(self, index):
        pcs_path = os.path.join(self.pcs, "{:06d}.bin".format(index))
        # 点云有四个数据(x, y, z, r)
        return np.fromfile(pcs_path, dtype=np.float32, count=-1).reshape([-1, 4])
 
    # 获取标签的方法
    def get_labels(self, index):
        labels_path = os.path.join(self.labels, "{:06d}.txt".format(index))
        with open(labels_path) as f:
            lines = f.readlines()
        # 去掉空行和换行
        lines = list(filter(lambda x: len(x) > 0 and x != '\n', lines))
        return [Object3d(x) for x in lines]




# 根据偏航角计算旋转矩阵（逆时针旋转）
def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return R


def project_to_image(pts_3d, P):
    
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]
    
def compute_box_3d(obj, P):
    # 得到旋转矩阵
    R = rot_y(obj.rotation_y)

    # 3d bounding box dimensions
    h, w, l = obj.dimensions[0], obj.dimensions[1], obj.dimensions[2]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # 得到目标物体经过旋转之后的实际尺寸（得到其在相机坐标系下的实际尺寸）
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    
    # 将该物体移动到相机坐标系下的原点处（涉及到坐标的移动，直接相加就行）
    corners_3d[0, :] += obj.location[0]
    corners_3d[1, :] += obj.location[1]
    corners_3d[2, :] += obj.location[2]
    

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def draw_projected_box3d(image, qs, color=(255, 0, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image
    
     
 

# 在rgb图像中显示2d的边界框
def bboxes_in_rgb(img, objects):
    img_2d, img_3d = img, img
    for obj in objects:
        # 为每个物体都要绘制2d矩形框
        cv2.rectangle(img_2d, (int(obj.bbox[0]), int(obj.bbox[1])), (int(obj.bbox[2]), int(obj.bbox[3])),
                      color=(0, 255, 0),
                      thickness=2)
    # 显示图像
    cv2.imshow("2d_bboxes", img_2d)
    return img_2d;
    
 # 在rgb图像中显示3d的边界框
def bboxes3d_in_rgb(img, objects, color=(255, 0, 255), thickness=2):

    for obj in objects:
        corners_2d, corners_3d = compute_box_3d(obj,calib.P2)
        img = draw_projected_box3d(img, corners_2d, color, thickness)
    cv2.imshow("3d_bboxes", img)

    return img;
















# 得到在前视图中的点云的索引
def get_lidar_index_in_image_fov(pcs, calib, xmin, ymin, xmax, ymax, clip_distance=2.0):
    # 先将点云转到rgb图像上来，然后保存[xmin,ymin,xmax,ymax]区域的像素（其实就是点云数据）
    # 然后选择正前方2m之外的点云数据
    # 返回索引
    pcs_2d = calib.project_vel_to_image(pcs.T)
    fov_pcs_indices = (
                (pcs_2d[0, :] >= xmin) & (pcs_2d[0, :] <= xmax) & (pcs_2d[1, :] >= ymin) & (pcs_2d[1, :] <= ymax))
    fov_pcs_indices = fov_pcs_indices & (pcs[:, 0] > clip_distance)
    return fov_pcs_indices

# 在点云图上显示3d的边界框
def bboxes_3d_in_pc(pcs, calib, fig=None, img_height=1224, img_width=370):
    import mayavi.mlab as mlab
    
    if fig is None: fig = mlab.figure(figure="point cloud", bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
    # 得到前视图的点云的索引
    pc_velo_index = get_lidar_index_in_image_fov(pcs[:, :3], calib, xmin=0, ymin=0, xmax=img_width, ymax=img_height)
    # 绘制点云
    mlab.points3d(pc_velo[:, 0], pc_velo[:, 1], pc_velo[:, 2], pc_velo[:, 2], mode="point", colormap="gnuplot",
                  scale_factor=0.3,
                  figure=fig)
    # 可视化点云
    mlab.show()
   
if __name__ == '__main__':
    # 用来可视化数据集中数据
    # 输入数据集的地址
    dir_path = '../dataset/'
    split = "training"
    dataset = Kitti_Dataset(dir_path, split=split)
    for i in range(len(dataset)):
    
        # 显示2D和3D的边界框处理
        img = dataset.get_rgb(i)
        objects = dataset.get_labels(i)
        calib = dataset.get_calib(i)
        
        img_2d = bboxes_in_rgb(img,objects)
        img_3d = bboxes3d_in_rgb(img,objects)
        
        # 点云
        pcs = dataset.get_pcs(i)
        pcs_2d = calib.project_vel_to_image(pcs[:, :3].T)
        cv2.imshow("3d_bboxes1", pcs_2d)
        #bboxes_3d_in_pc(pcs, calib)
        cv2.waitKey(5000)
     

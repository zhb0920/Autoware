#!/usr/bin/env python
import rospy
import math
import numpy as np
import chainer
import subprocess
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from autoware_msgs.msg import CloudCluster, CloudClusterArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
try:
    from voxelnet.config_utils import *
    from voxelnet.models import *
    from voxelnet.converter.voxelnet_concat import voxelnet_concat
    from voxelnet.models.light_voxelnet import LightVoxelnet
    import chainer.functions as F
    from chainer import dataset
    from chainer import serializers
    from data_util.kitti_util.input_velodyne import *
    from data_util.kitti_util.cython_util.create_input import *
except:
    pass


class PointCloudTestDataset(dataset.DatasetMixin):
    def __init__(self, pc, ignore_labels=True,
                 crop_size=(713, 713), color_sigma=None, g_scale=[0.5, 2.0],
                 resolution=None, x_range=None, y_range=None, z_range=None,
                 l_rotate=None, g_rotate=None, voxel_shape=None,
                 t=35, thres_t=0, norm_input=True,
                 anchor_size=(1.56, 1.6, 3.9), anchor_center=(-1.0, 0., 0.),
                 fliplr=False, n_class=19, scale_label=1):
        self.pc = pc
        self.ignore_labels = ignore_labels

        self.voxel_shape = voxel_shape
        self.resolution = resolution
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.anchor_size = anchor_size
        self.anchor_center = anchor_center
        self.scale_label = scale_label
        self.t = t
        self.thres_t = thres_t
        self.norm_input = norm_input

        self.proj_cam = None
        self.calib = None

    def __len__(self):
        return 1

    def get_example(self, i):
        pc = self.pc.copy()
        random_indexes = np.random.permutation(pc.shape[0])
        pc = pc[random_indexes]

        d, h, w = self.voxel_shape
        d_res, h_res, w_res = self.resolution
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range
        rotate = 0

        pc = np.ascontiguousarray(pc, dtype=np.float32)
        create_input = create_feature_input_rotate
        s = time.time()
        feature_input, counter, indexes, n_no_empty = \
            create_input(pc, d_res, h_res, w_res, self.t,
                         d, h, w, x_min, x_max, y_min, y_max, z_min, z_max,
                         self.thres_t, 0, 92)
        print("create input", time.time() - s)

        area_mask = create_mask(0, 90, d, h, w, self.scale_label).astype(np.int8)

        return (feature_input, counter, indexes,
                np.array([indexes.shape[0]]), np.array([n_no_empty]), area_mask)


def publisher_boudingboxes(reg):
    pub = rospy.Publisher("/bounding_boxes_car", BoundingBox, queue_size=1)
    r = rospy.Rate(10)
    theta = 0
    #boxes = BoundingBoxArray()
    box = BoundingBox()
    box.header.stamp = rospy.Time.now()
    box.header.frame_id = "velodyne"
    box.pose.orientation.x = 0
    box.pose.orientation.y = 0
    box.pose.orientation.z = math.sin(reg[0, 6]/2)
    box.pose.orientation.w = math.cos(reg[0, 6]/2)
    box.pose.position.x = reg[0, 0]
    box.pose.position.y = reg[0, 1]
    box.pose.position.z = reg[0, 2]
    box.dimensions.x = reg[0, 3]
    box.dimensions.y = reg[0, 4]
    box.dimensions.z = reg[0, 5]
    #boxes.boxes.append(box)
    #boxes.header.frame_id = "velodyne"
    #boxes.header.stamp = rospy.Time.now()
    pub.publish(box)
    r.sleep()


def callback(data):
    pc = np.empty((0, 4), np.float32)
    for p in pc2.read_points(data, field_names=("x", "y", "z", "intensity"), skip_nans=True):
        #print " x:%f y:%f z:%f i:%f" % (p[0], p[1], p[2], p[3])
        pc = np.append(pc, np.array([[p[0], p[1], p[2], p[3]]], np.float32), axis=0)
    test_data = PointCloudTestDataset(pc=pc, resolution=[0.4, 0.2, 0.2],
                                      x_range=[0, 70.4], y_range=[-40, 40],
                                      z_range=[-3, 1], voxel_shape=[10, 400, 352],
                                      scale_label=2, t=35, thres_t=0,
                                      anchor_size=(1.56, 1.6, 3.9),
                                      anchor_center=(-1.0, 0., 0.))
    test_iter = chainer.iterators.SerialIterator(test_data, repeat=False,
                                                 shuffle=False, batch_size=1)
    anchor = create_anchor(0.4, 0.2, 0.2, 10, 400, 352, -1.0,
                           0, -40, -3, 2)
    config = {"k": 3, "d": 10, "h": 400, "w": 352, "alpha": 1.5, "beta": 1.0, "p": 1}
    pretrained_model = {"download": False,
                        "path": "/home/daisuke/model.npz"}
                        #"path": "/home/daisuke/Autoware/ros/src/computing/perception/detection/lidar_detector/packages/lidar_voxelnet_detect/include/pretrained_model/model.npz"}
    model = LightVoxelnet(config, pretrained_model)
    gpu = -1
    if gpu != -1:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu(gpu)
    else:
        gpu = None
    for batch in test_iter:
        batch = voxelnet_concat(batch, gpu)
        result_reg, result_prob = model.inference(*batch,
                                                  anchor=anchor,
                                                  anchor_center=[-1.0, 0., 0.],
                                                  anchor_size=[1.56, 1.6, 3.9],
                                                  thres_prob=0.5,
                                                  nms_thresh=0.5)
        print(result_reg)
        print(result_prob)
        #print "x:{0} y:{1} z:{2} l:{3} w:{4} h:{5} r:{6}".format(result_reg[0, 0], result_reg[0, 1], result_reg[0, 2], result_reg[0, 3], result_reg[0, 4], result_reg[0, 5], result_reg[0, 6])
        publisher_boudingboxes(result_reg)

if __name__ == '__main__':
    rospy.init_node("lidar_voxelnet_detect")
    rospy.Subscriber("/points_raw", PointCloud2, callback, queue_size=1)
    rospy.spin()

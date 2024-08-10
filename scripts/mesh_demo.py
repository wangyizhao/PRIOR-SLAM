# -*- coding: utf-8 -*-
# 读取SLAM轨迹中的一段,为该段的每一个图像生成相应的per-frame Delaunay三角化图像和带有纹理的per-frame mesh
# 该文件所读取的数据格式仅适用于map_segment3
# 仅仅使用mesh.ply文件中存在的vertices来Delaunay三角剖分并生成新mesh.ply

import os
import cv2
import numpy as np
from scipy.spatial import Delaunay


def read_ply(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

        # 读取第5行第3个元素，并将其转化为整型
        vert_num = int(lines[4].strip().split()[2])

        # 读取第13行至（12+vert_num）行的数据，去掉每行的首尾空格后存入set中
        data_set = set()
        for line in lines[12:12 + vert_num]:
            data_set.add(line.strip())

    return data_set


# 读取单个图像并生成对应的per-frame Delaunay三角化图像
# im_id：关键帧ID
def process_image(im_id, input_image, KPs_file, MPs_file, input_dir, raw_dir, delaunay_dir, per_mesh_dir, ply_file):

    vert_set = read_ply(ply_file)

    # 读取关键点文件
    # 所在关键帧ID，关键点尺度，与关键点关联的地图点被观测次数，与关键点关联的地图点ID，关键点未校正x坐标，关键点未校正y坐标
    with open(KPs_file, 'r') as f:
        # build a dict: {与关键点关联的地图点ID：(关键点未校正x坐标，关键点未校正y坐标)}
        dict_MP2KP = {int(line.split()[3]): (int(line.split()[4]), int(line.split()[5]))
                      for line in f
                      if int(line.split()[0]) == im_id and int(line.split()[1]) <= 4}
    # 读取地图点文件
    # 地图点ID，地图点位置坐标x，y，z
    with open(MPs_file, 'r') as f:
        # build a dict: {地图点ID：(地图点位置坐标x，y，z)}
        dict_MP2Coord = {int(line.split()[0]): line.split()[1] + " " + line.split()[2] + " " + line.split()[3]
                         for line in f
                         if int(line.split()[0]) in dict_MP2KP}

    # 把value转换为列表
    KP_Coords = []
    MP_Coords = []
    for MP_ID in dict_MP2KP.keys():
        if MP_ID in dict_MP2Coord:
            vert = dict_MP2Coord[MP_ID]
            if vert in vert_set:
                KP_Coords.append(dict_MP2KP[MP_ID])
                MP_Coords.append((float(vert.split()[0]), float(vert.split()[1]), float(vert.split()[2])))

    if len(KP_Coords) <= 5:
        return

    KPs = np.array(KP_Coords)

    # all indexes of keypoints in Delaunay
    tri = Delaunay(KPs)

    # map point coords
    vertices = np.array(MP_Coords)
    num_vertices = len(vertices)

    # face indices
    faces = tri.simplices
    num_faces = len(faces)

    # 读取图像
    image = cv2.imread(input_dir + input_image)

    # 保存原图像
    cv2.imwrite(raw_dir + str(im_id) + ".png", image)
    cv2.imwrite(per_mesh_dir + str(im_id) + "_3.png", image)

    for ids in faces:
        pt0, pt1, pt2 = KP_Coords[ids[0]], KP_Coords[ids[1]], KP_Coords[ids[2]]
        cv2.line(image, pt0, pt1, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.line(image, pt0, pt2, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.line(image, pt1, pt2, (0, 255, 0), 1, lineType=cv2.LINE_AA)

        temp = ids[0]
        ids[0] = ids[2]
        ids[2] = temp

    for pt in KP_Coords:
        cv2.circle(image, pt, 2, (255, 0, 0), -1, lineType=cv2.LINE_AA)

    # 保存delaunay图像
    cv2.imwrite(delaunay_dir + "delaunay_" + str(im_id) + ".png", image)

    # 获取图像宽度和高度
    height, width, _ = image.shape
    up_to_scale = np.array([1.0 / float(width), -1.0 / float(height)])
    KPs = KPs * up_to_scale + np.array([0, 1.0])
    # face texture coords
    tex_coords = []
    for face in faces:
        tex_coords.append(
            (KPs[face[0]][0], KPs[face[0]][1], KPs[face[1]][0], KPs[face[1]][1], KPs[face[2]][0], KPs[face[2]][1]))

    # 生成PLY文件
    ply_header = '''ply
format ascii 1.0
comment VCGLIB generated
comment TextureFile {}
element vertex {}
property float x
property float y
property float z
element face {}
property list uchar int vertex_indices
property list uchar float texcoord
end_header
'''

    with open(per_mesh_dir + 'per_mesh_' + str(im_id) + '_3.ply', 'w') as file:

        file.write(ply_header.format(str(im_id) + "_3.png", num_vertices, num_faces))

        for vertex in vertices:
            file.write('{} {} {} \n'.format(vertex[0], vertex[1], vertex[2]))

        for i in range(len(faces)):
            file.write(
                '3 {} {} {} 6 {} {} {} {} {} {} \n'.format(faces[i][0], faces[i][1], faces[i][2], tex_coords[i][0],
                                                           tex_coords[i][1], tex_coords[i][2], tex_coords[i][3],
                                                           tex_coords[i][4], tex_coords[i][5]))


def main():

    seq_id = "1"
    KFs_file = "data" + seq_id + "/KeyFrames.txt"
    KPs_file = "data" + seq_id + "/KeyPoints.txt"
    MPs_file = "data" + seq_id + "/MapPoints_as.txt"
    input_dir = "/media/wyz/T7/Dataset/OpenLORIS-Scene/corridor1/corridor1-1/color/"
    raw_dir = "0_raw" + seq_id + "/"
    delaunay_dir = "1_delaunay" + seq_id + "/"
    per_mesh_dir = "2_per_mesh" + seq_id + "/"

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(delaunay_dir):
        os.makedirs(delaunay_dir)
    if not os.path.exists(per_mesh_dir):
        os.makedirs(per_mesh_dir)

    # 读取关键帧文件
    # 关键帧ID，图片名，关键帧位置坐标x，y，z，关键帧四元数q0，q1，q2，q3，共视关键帧ID
    with open(KFs_file, 'r') as f:
        # 关键帧字典 {关键帧ID：图片名}
        KFs_dict = {int(line.split()[0]): line.split()[1]
                    for line in f}

    for im_id in range(0, 10000):
        ply_file = "2_per_mesh" + seq_id + "/" + "per_mesh_" + str(im_id) + "_3.ply"
        if im_id in KFs_dict and os.path.exists(ply_file) and im_id == 5:
            print(im_id)
            process_image(im_id, KFs_dict[im_id], KPs_file, MPs_file, input_dir, raw_dir, delaunay_dir, per_mesh_dir, ply_file)

    print("Finished.")


if __name__ == '__main__':
    main()

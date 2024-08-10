# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import cKDTree
import cv2

from collections import defaultdict

# 读取深度图像
def read_depth_image(filename):
	depth_img = cv2.imread(filename, -1)  # uint16格式
	depth = depth_img.astype(np.float32) * 0.001  # 转换为米
	return depth

# 读取相机光心坐标
def read_groundtruth(filename):
	with open(filename, 'r') as f:
		line = f.readline().strip()
		values = line.split()
		timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, values)
		return np.array([tx, ty, tz]), np.array([qx, qy, qz, qw])

# 将深度图像转换为3D点云
def depth_to_3d_points(depth, K):
	fx, fy, cx, cy = K
	h, w = depth.shape

	# 创建网格坐标
	u = np.linspace(0, w-1, w)
	v = np.linspace(0, h-1, h)
	u, v = np.meshgrid(u, v)

	# 稀疏化加快运算
	for i in range(h):
		if i % 8 != 4:
			depth[i, :] = 0.0
		else:
			for j in range(w):
				if j % 8 != 4:
					depth[i, j] = 0.0

	# 使用内参计算3D坐标
	x = (u - cx) * depth / fx
	y = (v - cy) * depth / fy
	z = depth

	return np.stack((x, y, z), axis=-1)

def quaternion_to_rotation_matrix(q):
	qx, qy, qz, qw = q
	R = np.array([
		[1 - 2*(qy**2 + qz**2),	 2*(qx*qy - qz*qw),	 2*(qx*qz + qy*qw)],
		[	2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),	 2*(qy*qz - qx*qw)],
		[	2*(qx*qz - qy*qw),	 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
	])
	return R

# 将3D点云从相机坐标系转换到世界坐标系
def transform_to_world(points, R, t):
	points = points.reshape(-1, 3)
	
	# 过滤z坐标小于0.1的点
	valid_indices = points[:, 2] >= 0.1
	filtered_points = points[valid_indices]
	
	# 使用旋转矩阵和平移向量进行坐标转换
	transformed_points = np.dot(filtered_points, R.T) + t.T  # 使用广播
	
	return transformed_points

def read_ref_and_que_timestamps(filename):
	# 读取参考帧和查询帧的时间戳信息
	with open(filename, 'r') as f:
		ref_max_timestamp = float(f.readline().strip())
		que_min_timestamp = float(f.readline().strip())
	return ref_max_timestamp, que_min_timestamp

def load_depth_info(filename, ref_max_timestamp, que_min_timestamp):
	# 根据时间戳筛选出参考帧和查询帧的深度信息
	ref_depths = []
	que_depths = []
	with open(filename, 'r') as f:
		for line in f:
			timestamp, img_name = line.strip().split()
			timestamp = float(timestamp)
			if timestamp < ref_max_timestamp:
				ref_depths.append((timestamp, img_name))
			elif timestamp > que_min_timestamp:
				que_depths.append((timestamp, img_name))
	return ref_depths, que_depths

def find_closest_groundtruth(timestamp, groundtruth):
	# 查找时间戳最接近给定时间戳的真值信息
	closest_timestamp = min(groundtruth.keys(), key=lambda t: abs(t-timestamp))
	return groundtruth[closest_timestamp][0:3, 0:3], groundtruth[closest_timestamp][0:3, 3]

def load_groundtruth(filename):
	# 加载真值信息文件并将其存储为一个字典
	groundtruth = {}
	with open(filename, 'r') as f:
		for line in f:
			if line.startswith('#'):
				continue
			values = line.strip().split()
			timestamp = float(values[0])
			pose = np.array(list(map(float, values[1:])))
			groundtruth[timestamp] = pose
	return groundtruth

def load_rgb_dict(filename):

	rgb_dict = {}
	with open(filename, 'r') as f:
		for line in f:
			if line.startswith('#'):
				continue
			line = line.strip().split()
			rgb_dict[float(line[0])] = str(line[1])
	return rgb_dict

def load_transforms_from_yaml(filename):
	# 使用OpenCV加载YAML文件
	fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
	
	# 读取所有的转换矩阵
	transforms = {}
	trans_matrix_node = fs.getNode("trans_matrix")
	for i in range(trans_matrix_node.size()):
		node = trans_matrix_node.at(i)
		parent_frame = node.getNode("parent_frame").string()
		child_frame = node.getNode("child_frame").string()
		matrix_node = node.getNode("matrix")
		mat = matrix_node.mat()
		transforms[(parent_frame, child_frame)] = np.array(mat)
	
	return transforms

def transform_pose_through_frames(T, frames_sequence, transforms):
	# 应用转换序列以获得最终的位姿
	for i in range(len(frames_sequence) - 1):
		T = np.dot(np.dot(np.linalg.inv(transforms[frames_sequence[i], frames_sequence[i+1]]), T), transforms[frames_sequence[i], frames_sequence[i+1]])
	return T

def main():

	# 内参
	K = [6.1145098876953125e+02, 6.1148571777343750e+02, 4.3320397949218750e+02, 2.4947302246093750e+02]

	# 读取参考和查询帧的时间戳
	ref_max_timestamp, que_min_timestamp = read_ref_and_que_timestamps("ref_que.txt")
	# 加载参考帧和查询帧的深度信息
	ref_depths, que_depths = load_depth_info("aligned_depth.txt", ref_max_timestamp, que_min_timestamp)
	# 加载真值信息
	groundtruth = load_groundtruth("groundtruth.txt")


	# 加载所有的转换矩阵
	transforms = load_transforms_from_yaml('trans_matrix.yaml')
	# 定义转换序列
	frames_sequence = ["base_link", "d400_color_optical_frame", "d400_depth_optical_frame"]

	for timestamp, pose in groundtruth.items():
		T = np.eye(4)
		T[0:3, 0:3] = quaternion_to_rotation_matrix(pose[3:])
		T[0:3, 3] = pose[:3].reshape(-1)
		# 将根据base_link得到的真值位姿转换为d400_depth_optical_frame的位姿
		T = transform_pose_through_frames(T, frames_sequence, transforms)
		groundtruth[timestamp] = T

	print('finish load_groundtruth')



	# 转换参考帧的3D点云并存为cKDTree，方便后续查询
	ref_trees = {}
	# ref_groundtruth = {}
	ref_ave_points = []
	for timestamp, img_name in ref_depths:
		depth = read_depth_image(img_name)
		gt_R, gt_t = find_closest_groundtruth(timestamp, groundtruth)
		world_points = transform_to_world(depth_to_3d_points(depth, K), gt_R, gt_t)

		ave_point = np.mean(world_points, axis=0)
		ref_ave_points.append(ave_point)

		ref_trees[timestamp] = cKDTree(world_points)
		# ref_groundtruth[timestamp] = t
		print('	finish building tree ', timestamp)

	ref_ave_tree = cKDTree(ref_ave_points)
	print('finish building all trees')

	# gt_place = {}
	gt_place = defaultdict(set)

	err_th = 0.5
	in_ratio = 0.25
	shift = 6

	# 对每个查询帧进行匹配
	for timestamp, img_name in que_depths:
		depth = read_depth_image(img_name)
		gt_R, gt_t = find_closest_groundtruth(timestamp, groundtruth)
		world_points = transform_to_world(depth_to_3d_points(depth, K), gt_R, gt_t)

		ave_point = np.mean(world_points, axis=0)
		distance, index = ref_ave_tree.query(ave_point, k=1)


		for_index = index

		while True:
			if for_index < 0 or for_index >= len(ref_depths):
				break
			retrival_timestamp = ref_depths[for_index][0]
			distances, _ = ref_trees[retrival_timestamp].query(world_points, k=1)
			# 统计最近点小于err_th的3D点个数
			inliers = np.sum(distances < err_th)
			# 最近点小于err_th的3D点个数不少于所有3D点个数的in_ratio
			if inliers >= len(world_points) * in_ratio:
				gt_place[timestamp].add(retrival_timestamp)
				for_index = for_index + shift
			else:
				break


		back_index = index - shift

		while True:
			if back_index < 0 or back_index >= len(ref_depths):
				break
			retrival_timestamp = ref_depths[back_index][0]
			distances, _ = ref_trees[retrival_timestamp].query(world_points, k=1)
			# 统计最近点小于err_th的3D点个数
			inliers = np.sum(distances < err_th)
			# 最近点小于err_th的3D点个数不少于所有3D点个数的in_ratio
			if inliers >= len(world_points) * in_ratio:
				gt_place[timestamp].add(retrival_timestamp)
				back_index = back_index - shift
			else:
				break

		if timestamp in gt_place:
			print(timestamp, ref_depths[index][0], int(len(gt_place[timestamp])*shift/30))
		else:
			print(timestamp, 'no loop.')


	print('writing groundtruth_place')

	gt_place = sorted(gt_place.items(), key=lambda x: x[0])

	rgb_dict = load_rgb_dict('color.txt')

	# 将结果写入输出文件
	with open("groundtruth_place.txt", 'w') as f:
		for que_timestamp, retrivals in gt_place:

			rgb_timestamp = min(rgb_dict.keys(), key=lambda t: abs(t-que_timestamp))

			retrivals = sorted(retrivals, key=lambda x: x)
			retrival_min = retrivals[0]
			retrival_max = retrivals[-1]

			rgb_timestamp_min = min(rgb_dict.keys(), key=lambda t: abs(t-retrival_min))
			rgb_timestamp_max = min(rgb_dict.keys(), key=lambda t: abs(t-retrival_max))

			f.write("{} {} {}\n".format("%.6f" % rgb_timestamp, "%.6f" % rgb_timestamp_min, "%.6f" % rgb_timestamp_max))

	print('done')

if __name__ == '__main__':
	main()
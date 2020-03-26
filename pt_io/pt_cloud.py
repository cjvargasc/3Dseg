import numpy as np
import open3d as o3d
import random


class PointCloudIO:

    @staticmethod
    def read_pts(path):
        """ Reads the example file and returns a numpy array """
        f = open(path, "r")

        values = []
        for line in f.readlines():
            values.append(line.split(" "))
        f.close()
        values = np.array(values).astype(np.float)

        return values

    @staticmethod
    def visualization_tests(values):
        """ Shows the point cloud in path file """

        xyz = np.zeros((values.shape[0], 3))
        xyz[:, 0] = np.reshape(values[:, 0], -1)
        xyz[:, 1] = np.reshape(values[:, 1], -1)
        xyz[:, 2] = np.reshape(values[:, 2], -1)

        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Set True to visualize GT boundaries
        if False:
            print("GT boundaries visualization")
            colors = np.zeros((values.shape[0], 3))
            colors[values[:, 6] == 1] = (1, 0, 0)
            colors[values[:, 6] == 0] = (0.5, 0.9, 0.85)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        if True:
            threshold = 0.994
            print("Boundaries visualization, t=", threshold)
            colors = np.zeros((values.shape[0], 3))
            colors[values[:, 7] >= threshold] = (1, 0, 0)
            colors[values[:, 7] < threshold] = (0.5, 0.9, 0.85)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])

    @staticmethod
    def show_segments(values, labels):

        random.seed(123)

        xyz = np.zeros((values.shape[0], 3))
        xyz[:, 0] = np.reshape(values[:, 0], -1)
        xyz[:, 1] = np.reshape(values[:, 1], -1)
        xyz[:, 2] = np.reshape(values[:, 2], -1)

        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Set True to visualize GT boundaries
        colors = np.zeros((values.shape[0], 3))

        for segment in np.unique(labels):

            if segment == -1:  # boundaries
                random_r = 1
                random_g = 0
                random_b = 0
            else:
                random_r = random.random()
                random_g = random.random()
                random_b = random.random()

            colors[labels == segment] = (random_r, random_g, random_b)

        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])

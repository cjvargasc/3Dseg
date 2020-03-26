import numpy as np
from pt_io.pt_cloud import PointCloudIO
from utils.misc import Utils
from seg.ex1 import Segmenter
from seg.ex2 import NcutSegmenter


def exercise1():
    pts_file = "example.pts"
    selected_thresh = 0.994

    k = 5

    print("** Loading point cloud... **")
    points = PointCloudIO.read_pts(pts_file)
    print("Done.")

    print("** Point cloud visualization **")
    PointCloudIO.visualization_tests(points)

    Utils.print_boundaries_stats(points)

    Utils.f1score_thresh_plot(points)

    print("Selected threshold: ", selected_thresh)
    print("f1-score", Utils.calc_f1score(points, selected_thresh))

    print("")
    print("ex1: Segmentation")
    seg = Segmenter(points, selected_thresh)
    segments = seg.segment(k)
    print("Done..")

    print(np.unique(segments))
    print("Segments found: ", np.unique(segments).shape[0] - 1)  # minus boundaries

    PointCloudIO.show_segments(points, segments)


def exercise2():

    pts_file = "example.pts"
    k = 5

    print("** Loading point cloud... **")
    points = PointCloudIO.read_pts(pts_file)
    print("Done.")

    seg = NcutSegmenter(points, k)

    print("Ncuts: Normal angle weighted segmentation")
    segments = seg.segment_func1()

    print(np.unique(segments))
    print("Segments found: ", np.unique(segments).shape[0])

    PointCloudIO.show_segments(points, segments)

    print("Ncuts: boundary probabilities weighted segmentation")
    segments = seg.segment_func1()

    print(np.unique(segments))
    print("Segments found: ", np.unique(segments).shape[0])

    PointCloudIO.show_segments(points, segments)


if __name__ == "__main__":
    exercise1()
    exercise2()

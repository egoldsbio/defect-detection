import numpy as np
from scipy.stats import mode

def update_centroid(pcd):
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    pcd.translate(-centroid)
    return pcd

def filter_background(pcd, recenter=True):
    pt_to_pln_dist = 0.002

    _, ind = pcd.segment_plane(distance_threshold = pt_to_pln_dist, ransac_n = 3, num_iterations = 1000)
    inlier_cloud = pcd.select_by_index(ind, invert=True)
    
    labels = np.array(inlier_cloud.cluster_dbscan(eps = 0.005, min_points = 1))
    max_label = mode(labels).mode
    largest_cluster = np.where(labels == max_label)[0]

    main_pcd = inlier_cloud.select_by_index(largest_cluster)

    if recenter:
        main_pcd = update_centroid(main_pcd)

    return main_pcd
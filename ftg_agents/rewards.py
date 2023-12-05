import numpy as np
from typing import Tuple

class Progress:
    def __init__(self, track, lookahead: int = 20) -> None:
        # 
        xs = track.centerline.xs
        ys = track.centerline.ys
        self.centerline = np.stack((xs, ys), axis=-1)
        # append first point to end to make loop
        self.centerline = np.vstack((self.centerline, self.centerline[0]))

        self.segment_vectors = np.diff(self.centerline, axis=0)
        # print(segment_vectors.shape)
        self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
        
        # Extend segment lengths to compute cumulative distance
        self.cumulative_lengths = np.hstack(([0], np.cumsum(self.segment_lengths)))
        self.previous_closest_idx = 0
        self.max_lookahead = lookahead
        #print(self.centerline)
        #print(self.centerline.shape)
        #print("***********")

    def distance_along_centerline_np(self, pose_points):
        assert len(pose_points.shape) == 2 and pose_points.shape[1] == 2

        # centerpoints = np.array(centerpoints)
        #print(self.centerline.shape)
        #print(centerpoints[:-1])
        #print(pose_points)
        #print(".....")
        # assert pose points must be Nx2
        pose_points = np.array(pose_points)
        #print(pose_points.shape)
        #print(pose_points)
        #print("distance calc")
        #print(self.previous_closest_idx)
        def projected_distance(pose):
            rel_pose = pose - self.centerline[:-1]
            t = np.sum(rel_pose * self.segment_vectors, axis=1) / np.sum(self.segment_vectors**2, axis=1)
            t = np.clip(t, 0, 1)
            projections = self.centerline[:-1] + t[:, np.newaxis] * self.segment_vectors
            distances = np.linalg.norm(pose - projections, axis=1)
            points_len = self.centerline.shape[0]-1  # -1 because of last fake 
            lookahead_idx = (self.max_lookahead + self.previous_closest_idx) % points_len
            # wrap around
            if self.previous_closest_idx <= lookahead_idx:
                indices_to_check = list(range(self.previous_closest_idx, lookahead_idx + 1))
            else:
                # Otherwise, we need to check both the end and the start of the array
                indices_to_check = list(range(self.previous_closest_idx, points_len)) \
                    + list(range(0, lookahead_idx+1))
            # Extract the relevant distances using fancy indexing
            subset_distances = distances[indices_to_check]

            # Find the index of the minimum distance within this subset
            subset_idx = np.argmin(subset_distances)

            # Translate it back to the index in the original distances array
            closest_idx = indices_to_check[subset_idx]
            self.previous_closest_idx = closest_idx
            # print(closest_idx)
            return self.cumulative_lengths[closest_idx] + self.segment_lengths[closest_idx] * t[closest_idx]
        
        return np.array([projected_distance(pose) for pose in pose_points])
    
    # TODO is this not wrong? (the tuple)
    def get_progress(self, pose: Tuple[float, float]):
        #print("---get")
        #print(pose)
        progress =  self.distance_along_centerline_np(pose)
        # print(self.cumulative_lengths.shape)
        # print(self.cumulative_lengths[-1])
        progress = progress / (self.cumulative_lengths[-1] + self.segment_lengths[-1])
        # clip between 0 and 1 (it can sometimes happen that its slightly above 1)
        # print("progress", progress)
        return np.clip(progress, 0, 1)
    # input shape: tuple of (x,y)
    def reset(self, pose):
        rel_pose = pose - self.centerline[:-1]
        t = np.sum(rel_pose * self.segment_vectors, axis=1) / np.sum(self.segment_vectors**2, axis=1)
        t = np.clip(t, 0, 1)
        projections = self.centerline[:-1] + t[:, np.newaxis] * self.segment_vectors
        distances = np.linalg.norm(pose - projections, axis=1)
        
        closest_idx = np.argmin(distances)
        self.previous_closest_idx = closest_idx

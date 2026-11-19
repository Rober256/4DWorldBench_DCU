import torch
import lietorch
import numpy as np

from lietorch import SE3
from .factor_graph import FactorGraph


class DroidBackend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.upsample = args.upsample
        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms
        
    @torch.no_grad()
    def __call__(self, steps=12):
        """ main update """

        t = self.video.counter.value
        if not self.video.stereo and not torch.any(self.video.disps_sens):
             self.video.normalize()

        graph = FactorGraph(self.video, self.update_op, corr_impl="volume", max_factors=16*t, upsample=self.upsample)

        graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, 
                                    beta=self.beta)

        for i in range(steps):
            print("Global BA Iteration #{}".format(i+1))
            graph.update(t0=1, t1=t, itrs=2, motion_only=False)

        # Compute reprojection errors after BA
        with torch.no_grad():
            coords1, valid_mask = self.video.reproject(graph.ii, graph.jj)
            residual = (graph.target - coords1).norm(dim=-1)
            residual_flat = residual.reshape(-1)
            valid_flat = valid_mask.reshape(-1) > 0.5
            if valid_flat.any():
                valid_errors = residual_flat[valid_flat]
            else:
                valid_errors = residual_flat

        graph.clear_edges()
        self.video.dirty[:t] = True
        return valid_errors

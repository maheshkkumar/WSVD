import cv2
import numpy as np
import torch

from .networks.flownet2 import load_flownet, pred_flo
from .networks.resdepth import TempDepthNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DepthPred():
    def __init__(self,
                 imH=384,
                 imW=704,
                 ckpt='ckpts/yt3d_tempdepth_2frame/model_cur'):
        
        self.depth_net = TempDepthNet(imH, imW)
        self.depth_net.load_model(ckpt)
        self.depth_net.to(device)
        self.depth_net.eval()

        self.flow_net = load_flownet()
        self.flow_net.eval()
        self.flow_net.to(device)

        self.imH = imH
        self.imW = imW

    def pred(self, im1_np, im2_np):
        # resize image to match the training set stat.
        im1_np = cv2.resize(im1_np, (self.imW, self.imH))
        im2_np = cv2.resize(im2_np, (self.imW, self.imH))
        # 1. predict flow with flownet2.0
        flo_np = pred_flo(self.flow_net, [im1_np], [im2_np], 0)[0]

        # 2. feed 2 images and the flow to depth estimator
        im1 = torch.from_numpy(np.transpose(
            im1_np, [2, 0, 1]).astype(np.float32))
        im2 = torch.from_numpy(np.transpose(
            im2_np, [2, 0, 1]).astype(np.float32))
        flo = torch.from_numpy(flo_np)
        with torch.no_grad():
            im1 = im1.unsqueeze(0).to(device)/255
            im2 = im2.unsqueeze(0).to(device)/255
            flos = flo.unsqueeze(0).to(device)
            log_depth_pred = self.depth_net.forward(im1, im2, flos)
            pred_d = log_depth_pred.exp()

        pred_d = pred_d.data.cpu().numpy()[0, ...]
        return pred_d

class WSVD():
    def __init__(self):
        self.weights = './models/WSVD/weights/wsvd.pth'
        self.model = DepthPred(ckpt=self.weights)

    def evaluate(self, img):
        img = cv2.imread(img)[..., ::-1]
        pred_d = self.model.pred(img[0], img[1])

        # convert depth to inverse depth
        pred_q = 1.0 / pred_d

        # min max normalization
        pred_q = (pred_q - pred_q.min()) / (pred_q.max() - pred_q.min())

        return pred_q

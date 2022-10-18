import os
import numpy as np
import cv2
from glob import glob
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    #flow_frame = np.clip(flow_frame, -20, 20)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow

def cal_for_frames(video_path):
    print(video_path)
    frames = glob(os.path.join(video_path, '*.png'))
    frames.sort()
    #print(frames)
    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow
# if opencv is using GPU?

def save_flow(video_flows, flow_path):
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path.format('u'), "{}_{:06d}.jpg".format(obj_id, i)), flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path.format('v'), "{}_{:06d}.jpg".format(obj_id, i)), flow[:, :, 1])
        
def extract_flow(args):
    video_path, flow_path = args
    flow = cal_for_frames(video_path)
    save_flow(flow, flow_path)
    #print('complete:' + flow_path)
    return
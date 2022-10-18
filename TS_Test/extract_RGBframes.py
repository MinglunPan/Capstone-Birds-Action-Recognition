import os
import numpy as np
import cv2
from glob import glob
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler


def cal_for_RGB_frames(video_path):
    print(video_path)
    frames = glob(os.path.join(video_path, '*.png'))
    frames.sort()
    #print(frames)
    flow = []
    for f in frames:
        rgb_frame = cv2.imread(f)
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
        flow.append(rgb_frame)

    return flow

def save_flow(video_flows, flow_path):
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path.format('u'), "{}_{:06d}.jpg".format(obj_id, i)), flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path.format('v'), "{}_{:06d}.jpg".format(obj_id, i)), flow[:, :, 1])
        
def extract_RGBframes(args):
    video_path, flow_path = args
    flow = cal_for_RGB_frames(video_path)
    save_flow(flow, flow_path)
    #print('complete:' + flow_path)
    return
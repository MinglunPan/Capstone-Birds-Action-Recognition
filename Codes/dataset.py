from config import IMAGE_METADATA_PATH, OBJ_ID_UNIQUE_COLUMNS, OBJ_ID_MULTI_COLUMNS
import pandas
import os
import config
import numpy as np
import tensorflow as tf
import image

def load_images_metadata():
    data = pandas.read_csv(IMAGE_METADATA_PATH)
    data.sort_values(by = ['obj_id','frame'], inplace = True)
    return data

def load_images_metadata_dict(data):
    
    metadata_col_idx_dict = {col:idx for idx,col in enumerate(data.columns)}
    images_metadata_dict = dict()
    for row in data.values:
        obj_id = row[metadata_col_idx_dict.get('obj_id')]

        if obj_id not in images_metadata_dict:
            images_metadata_dict[obj_id] = {
                col:row[metadata_col_idx_dict.get(col)]
                        for col in OBJ_ID_UNIQUE_COLUMNS}
            for col in OBJ_ID_MULTI_COLUMNS:
                images_metadata_dict[obj_id][col] = []
        for col in OBJ_ID_MULTI_COLUMNS:
            images_metadata_dict[obj_id][col].append( row[metadata_col_idx_dict.get(col)] )
    return images_metadata_dict

def configure_for_performance(tf_dataset):
    return tf_dataset.batch(config.BATCH_SIZE)


def generator_dataset_activity(obj_metadata_dict):
    for obj_id, obj_info in obj_metadata_dict.items():
  
        data_path = os.path.join(*[
            str(x) for x in [config.DATA_DIR, obj_info['day_dir'], obj_info['camera_dir'], 
                             obj_info['video_dir'], obj_info['track_dir']]])
        frame_path_list = [os.path.join(data_path, file_path) for file_path in obj_info['image_file']]
        # Counter
        num_total_frames = len(frame_path_list) # the number of imgs in this track
        num_used_frames = min(config.NUM_INPUT_FRAME, num_total_frames)
        num_blank_frames = config.NUM_INPUT_FRAME - num_used_frames
        num_skip_at_begining = max(0, (num_total_frames-config.NUM_INPUT_FRAME) >> 1)

        # Frames
        frames = np.zeros([config.NUM_INPUT_FRAME, config.IMG_HEIGHT,  config.IMG_WIDTH,  config.NUM_CHANNEL])
        frames[:num_used_frames] = tf.image.resize(
            np.stack([
                image.load(frame_path_list[num_skip_at_begining+frame_idx]) 
                for frame_idx in range(num_used_frames)]
            ), [config.IMG_HEIGHT, config.IMG_WIDTH])
        
        frames = tf.convert_to_tensor(frames, tf.float32)
        # Labels
        label = obj_info['activity_cat']
        yield frames, label
        

def generator_dataset(obj_metadata_dict, format = (('input_1', 'input_2'), 'label')):
    for obj_id, obj_info in obj_metadata_dict.items():
  
        data_path = os.path.join(*[
            str(x) for x in [config.DATA_DIR, obj_info['day_dir'], obj_info['camera_dir'], 
                             obj_info['video_dir'], obj_info['track_dir']]])
        frame_path_list = [os.path.join(data_path, file_path) for file_path in obj_info['image_file']]
        # Counter
        num_total_frames = len(frame_path_list) # the number of imgs in this track
        num_used_frames = min(config.NUM_INPUT_FRAME, num_total_frames)
        num_blank_frames = config.NUM_INPUT_FRAME - num_used_frames
        num_skip_at_begining = max(0, (num_total_frames-config.NUM_INPUT_FRAME) >> 1)
        # Mark Tensor
        mark_tensor = np.zeros([config.NUM_INPUT_FRAME,])
        mark_tensor[:num_used_frames] = 1
        mark_tensor = mark_tensor.reshape([config.NUM_INPUT_FRAME, 1, 1, 1])
        mark_tensor = tf.convert_to_tensor(mark_tensor, tf.float32)
        # Frames
        frames = np.zeros([config.NUM_INPUT_FRAME, config.IMG_HEIGHT,  config.IMG_WIDTH,  config.NUM_CHANNEL])
        frames[:num_used_frames] = np.stack([image.load(frame_path_list[num_skip_at_begining+frame_idx]) 
                         for frame_idx in range(num_used_frames)])

        frames = tf.convert_to_tensor(frames, tf.float32)
        # Labels
        label = obj_info['obj_cat_binary']
        yield ({"input_1": frames, "input_2": mark_tensor}, label) 
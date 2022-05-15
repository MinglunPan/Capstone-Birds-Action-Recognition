import tensorflow as tf

IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CHANNEL = 3
NUM_INPUT_FRAME = 40
NUM_ACT_CAT = 8
BATCH_SIZE = 16
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)


DATA_DIR = "/project2/msca/projects/AvianSolar/ImageDataset/raw_dataset"
IMAGE_METADATA_PATH = "/project2/msca/projects/AvianSolar/ImageDataset/raw_dataset/all_image_merged.csv"

OBJ_LIST = ['bird', 'cable', 'panel', 'plant', 'car', 'human', 
            'other_animal', 'insect', 'aircraft', 'other', 'unknown']
ACT_LIST = ['fly_over_above', 'fly_over_reflection', 'fly_through', 
            'perch_on_panel', 'land_on_ground', 'perch_in_background',
            'collision','uncertain',]

OBJ_ID_UNIQUE_COLUMNS = ['day_dir', 'camera_dir', 'video_dir', 'track_dir', 'directory_x',
       'count', 'id', 'directory_y', 'bird', 'cable', 'panel', 'plant', 'car',
       'human', 'other_animal', 'insect', 'aircraft', 'other', 'unknown',
       'fly_over_above', 'fly_over_reflection', 'fly_through',
       'perch_on_panel', 'land_on_ground', 'perch_in_background', 'collision',
       'uncertain', 'image_count', 'obj_cat', 'obj_cat_binary', 'activity_cat',
       'ttv_split']
OBJ_ID_MULTI_COLUMNS = ['image_file','x','y','speed','area']


X_SHAPE = (None,IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL)
DATASET_SHAPE = (X_SHAPE,())
DATASET_TYPE = (tf.float32, tf.int64)
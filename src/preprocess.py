import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
import glob
from tqdm import tqdm
import yaml
import zipfile
from src.utils import load_video, char_to_num

def load_alignments(path: str) -> List[str]:
    """Loads and tokenizes alignment files."""
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens.extend(list(line[2]))
            tokens.append(' ')
    # Remove the trailing space
    if tokens:
        tokens.pop()
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, 'UTF-8'), (-1)))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tf_example(video_path, alignment_path):
    """Creates a tf.train.Example proto from video and alignment."""
    frames = load_video(video_path)
    if frames is None:
        return None
    alignments = load_alignments(alignment_path)
    
    feature = {
        'frames': _bytes_feature(tf.io.serialize_tensor(frames)),
        'alignments': _bytes_feature(tf.io.serialize_tensor(alignments))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def main():
    """Main preprocessing function to create TFRecords."""
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    data_params = params['data']
    raw_dir = data_params['raw_dir']
    zip_path = os.path.join(raw_dir, data_params['zip_file'])
    tfrecord_path = data_params['tfrecord_file']
    
    os.makedirs(os.path.dirname(tfrecord_path), exist_ok=True)

    extract_dir = os.path.join(raw_dir, 'extracted_data')
    if not os.path.exists(os.path.join(extract_dir, 'data', 's1')):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete.")
    else:
        print("Data already extracted.")

    video_paths = glob.glob(os.path.join(extract_dir, 'data', 's1', '*.mpg'))
    
    print(f"Found {len(video_paths)} videos to process.")

    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for video_path in tqdm(video_paths, desc="Processing videos"):
            file_name = os.path.basename(video_path).split('.')[0]
            alignment_path = os.path.join(extract_dir, 'data', 'alignments', 's1', f'{file_name}.align')
            
            if not os.path.exists(alignment_path):
                print(f"Warning: Alignment file not found for {video_path}. Skipping.")
                continue

            try:
                tf_example = create_tf_example(video_path, alignment_path)
                if tf_example:
                    writer.write(tf_example.SerializeToString())
            except Exception as e:
                print(f"Skipping {video_path} due to error: {e}")

    print(f"TFRecord file created at {tfrecord_path}")

if __name__ == '__main__':
    main()

import tensorflow as tf
import cv2
from typing import List

# This vocabulary must be consistent across preprocessing, training, and prediction.
VOCAB = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Create the character-to-number mapping
char_to_num = tf.keras.layers.StringLookup(vocabulary=VOCAB, oov_token="")

# Create the number-to-character mapping
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path: str) -> List[float]:
    """
    Loads and preprocesses a single video file.
    This function must be identical to the one used during training.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = tf.image.rgb_to_grayscale(frame)
        # Crop to the mouth region
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    if not frames:
        return None

    # Normalize the frames
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    normalized_frames = tf.cast((frames - mean), tf.float32) / std
    
    return normalized_frames

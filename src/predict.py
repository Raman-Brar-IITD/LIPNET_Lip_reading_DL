import tensorflow as tf
from src.utils import load_video, num_to_char

def predict_video(model, video_path: str) -> str:
    """
    Takes a Keras model and a video path, and returns the predicted text.
    """
    # 1. Load and preprocess the video
    video_frames = load_video(video_path)
    if video_frames is None:
        return "Could not process video. Check if it's a valid file."

    # 2. Add a batch dimension to match model's expected input shape
    video_frames = tf.expand_dims(video_frames, axis=0)
    
    # 3. Make a prediction
    yhat = model.predict(video_frames, verbose=0)
    
    # 4. Decode the prediction using CTC decode
    # The 'input_length' is the number of time steps from the model's output
    input_length = [yhat.shape[1]] 
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=input_length, greedy=True)[0][0].numpy()
    
    # 5. Convert from numbers to characters and join them
    predicted_text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8')
    
    return predicted_text

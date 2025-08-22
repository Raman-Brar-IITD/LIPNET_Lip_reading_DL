import os
import subprocess
import time
import tensorflow as tf
import yaml
from flask import Flask, render_template, request, jsonify

# Import functions and variables from your project's 'src' directory
from src.predict import predict_video
from src.train import build_model
from src.utils import char_to_num

# --- 1. INITIAL SETUP ---
app = Flask(__name__)

# --- Using the correct path you found via the 'where' command ---
FFMPEG_PATH = r"C:\Users\Raman\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"


# --- 2. LOAD THE TRAINED MODEL (Done once at startup) ---
print("Loading lip-reading model...")
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Get model and training parameters
model_params = params['model']
train_params = params['train']
input_shape = model_params['input_shape']
vocab_size = len(char_to_num.get_vocabulary())
model_weights_path = train_params['checkpoint_path']

# Build the model architecture
model = build_model(input_shape, vocab_size)

# Load the trained weights
if os.path.exists(model_weights_path):
    model.load_weights(model_weights_path)
    print(f"Model weights loaded successfully from {model_weights_path}.")
else:
    print(f"WARNING: Model weights not found at {model_weights_path}. Predictions will be random.")

# --- 3. VIDEO DATA DIRECTORIES ---
ORIGINAL_VIDEO_DIR = os.path.join('data', 'raw', 'extracted_data', 'data', 's1')
CONVERTED_VIDEO_DIR = 'static/converted_videos'
os.makedirs(CONVERTED_VIDEO_DIR, exist_ok=True)


def get_original_video_files():
    """Returns a sorted list of all .mpg files from the original dataset."""
    if not os.path.exists(ORIGINAL_VIDEO_DIR):
        print(f"WARNING: Original video directory not found at {ORIGINAL_VIDEO_DIR}.")
        return []
    return sorted([f for f in os.listdir(ORIGINAL_VIDEO_DIR) if f.endswith('.mpg')])

# --- 4. VIDEO CONVERSION FUNCTION ---
def convert_to_mp4_for_playback(original_video_path):
    """
    Converts the selected video to a temporary MP4 file for web playback.
    This file is always named 'selected_video.mp4' and is overwritten each time.
    Returns the path to the converted file.
    """
    output_path = os.path.join(CONVERTED_VIDEO_DIR, 'selected_video.mp4')
    
    print(f"Converting {os.path.basename(original_video_path)} for web playback...")
    
    cmd = [
        FFMPEG_PATH, '-i', original_video_path, '-y',
        '-vcodec', 'libx264', '-acodec', 'aac', output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Successfully created temporary playback file at {output_path}.")
        return output_path
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: FFmpeg conversion failed. Error: {e}")
        print(f"Attempted to use FFmpeg from: {FFMPEG_PATH}")
        return None

# --- 5. FLASK ROUTES ---
@app.route('/')
def index():
    """Renders the main page with the list of original videos."""
    return render_template('index.html', video_files=get_original_video_files())

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction logic. This is called by the JavaScript fetch API.
    It converts the video, runs the prediction, and returns the result as JSON.
    """
    data = request.get_json()
    selected_video_filename = data.get('video_filename')

    if not selected_video_filename:
        return jsonify({'error': 'No video filename provided.'}), 400

    original_video_path = os.path.join(ORIGINAL_VIDEO_DIR, selected_video_filename)

    if not os.path.exists(original_video_path):
        return jsonify({'error': 'Video file not found.'}), 404

    # Convert the video to the temporary file
    converted_video_path = convert_to_mp4_for_playback(original_video_path)
    if not converted_video_path:
        return jsonify({'error': 'Failed to convert video for playback.'}), 500

    try:
        # Run prediction on the original video file
        prediction_text = predict_video(model, original_video_path)
        
        # Create a cache buster using the current time
        cache_buster = int(time.time())
        
        # Return the result as JSON
        return jsonify({
            'prediction': prediction_text,
            'video_to_play': 'selected_video.mp4',
            'cache_buster': cache_buster
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500


# --- 6. START THE SERVER ---
if __name__ == '__main__':
    print(f"Starting Flask server on http://127.0.0.1:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)

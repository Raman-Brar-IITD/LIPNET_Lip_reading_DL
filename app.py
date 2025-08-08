import os
import cv2
import yaml
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Import functions and variables from your project's 'src' directory
from src.train import build_model
from src.predict import predict_frames
from src.utils import VOCAB

# --- 1. INITIAL SETUP ---
app = Flask(__name__)
# A secret key is required by Flask-SocketIO for session management
app.config['SECRET_KEY'] = 'your-very-secret-key'
# Initialize SocketIO for real-time communication
socketio = SocketIO(app)

# --- 2. LOAD THE TRAINED MODEL (Done once at startup) ---
print("Loading lip-reading model...")
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Get model and training parameters from params.yaml
model_params = params['model']
train_params = params['train']
input_shape = model_params['input_shape']
vocab_size = len(VOCAB)
model_weights_path = train_params['checkpoint_path']

# Build the model architecture (reusing the function from train.py)
model = build_model(input_shape, vocab_size)

# Load the trained weights into the model structure
if os.path.exists(model_weights_path):
    model.load_weights(model_weights_path)
    print(f"Model weights loaded successfully from {model_weights_path}.")
else:
    print(f"WARNING: Model weights not found at {model_weights_path}. Predictions will be random.")

# --- 3. FRAME BUFFER ---
# This list will act as a temporary buffer to hold frames from the webcam stream
frame_buffer = []

# --- 4. FLASK & SOCKETIO ROUTES ---
@app.route('/')
def index():
    """Renders the main HTML page for the application."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """
    This function is called when a new client connects to the server.
    It sends a confirmation message back to the client.
    """
    print('Client connected')
    emit('status', {'msg': 'Connected. You can start streaming.'})

@socketio.on('image')
def handle_image(data_image):
    """
    This function receives and processes each video frame sent from the client.
    """
    global frame_buffer
    
    # Decode the base64 image data received from the browser
    sbuf = base64.b64decode(data_image.split(',')[1])
    nparr = np.frombuffer(sbuf, dtype=np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess the frame to match the format used during training
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    frame = frame[190:236, 80:220] # Crop to the mouth region
    
    # Add the processed frame to our buffer
    frame_buffer.append(frame)

    # When the buffer reaches the required sequence length (75 frames),
    # it's time to run a prediction.
    if len(frame_buffer) == 75:
        try:
            # Convert the buffer to a NumPy array for the model
            frames_to_predict = np.array(frame_buffer)
            
            # Normalize the frames just like in training
            mean = np.mean(frames_to_predict)
            std = np.std(frames_to_predict)
            frames_to_predict = (frames_to_predict - mean) / std
            
            # Run the prediction
            prediction = predict_frames(model, frames_to_predict)
            
            # Send the resulting text back to the client
            emit('response', {'data': prediction})
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            emit('response', {'data': '[Error]'})
            
        finally:
            # Clear the buffer to start collecting the next 75-frame sequence
            frame_buffer.clear()

@socketio.on('disconnect')
def handle_disconnect():
    """
    This function is called when a client disconnects.
    It clears the buffer to ensure a clean state for the next connection.
    """
    global frame_buffer
    frame_buffer.clear()
    print('Client disconnected')

# --- 5. START THE SERVER ---
if __name__ == '__main__':
    # Use socketio.run() to start the web server.
    # 'eventlet' is a recommended server for production use with SocketIO.
    print("Starting Flask-SocketIO server on http://127.0.0.1:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)

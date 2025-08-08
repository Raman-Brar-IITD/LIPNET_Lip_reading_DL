import os
import gdown
import tensorflow as tf
import yaml

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv3D, LSTM, Dense, Dropout, Bidirectional,
                                     MaxPool3D, Reshape)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from src.utils import VOCAB

# --- VOCABULARY SETUP ---
char_to_num = tf.keras.layers.StringLookup(vocabulary=VOCAB, oov_token="")

def parse_tfrecord_fn(example):
    feature_description = {
        'frames': tf.io.FixedLenFeature([], tf.string),
        'alignments': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_description)
    frames = tf.io.parse_tensor(example['frames'], out_type=tf.float32)
    alignments = tf.io.parse_tensor(example['alignments'], out_type=tf.int64)
    return frames, alignments

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones((batch_len, 1), dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones((batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def scheduler(epoch, lr):
    return lr if epoch < 1 else lr * tf.math.exp(-0.1)

def build_model(input_shape, vocab_size):
    return Sequential([
        Conv3D(128, 3, input_shape=input_shape, padding='same', activation='relu'),
        MaxPool3D((1, 2, 2)),
        Conv3D(256, 3, padding='same', activation='relu'),
        MaxPool3D((1, 2, 2)),
        Conv3D(75, 3, padding='same', activation='relu'),
        MaxPool3D((1, 2, 2)),
        Reshape((input_shape[0], -1)),
        Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='Orthogonal')),
        Dropout(0.5),
        Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='Orthogonal')),
        Dropout(0.5),
        Dense(vocab_size + 1, activation='softmax', kernel_initializer='he_normal')
    ])

def main():
    # --- Load config ---
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    # --- Parameters ---
    tfrecord_path = params['data']['tfrecord_file']
    train_params = params['train']
    model_params = params['model']
    
    input_shape = model_params['input_shape']
    padded_shapes = (
        model_params['padded_shapes']['frames'],
        model_params['padded_shapes']['alignments']
    )

    checkpoint_path = train_params['checkpoint_path']
    weight_url = train_params["weight_url"]
    weight_local = "models/wght.weights.h5"
    
    epochs = train_params['epochs']
    batch_size = train_params['batch_size']
    learning_rate = train_params['learning_rate']

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # --- GPU Memory Growth ---
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # --- Dataset ---
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(buffer_size=1000)

    dataset_size = sum(1 for _ in dataset)
    train_size = int(0.9 * dataset_size)
    
    train_data = dataset.take(train_size)
    test_data = dataset.skip(train_size)

    train_pipeline = train_data.padded_batch(batch_size, padded_shapes=padded_shapes).prefetch(tf.data.AUTOTUNE)
    test_pipeline = test_data.padded_batch(batch_size, padded_shapes=padded_shapes).prefetch(tf.data.AUTOTUNE)

    # --- Build Model ---
    vocab_size = char_to_num.vocabulary_size()
    model = build_model(input_shape, vocab_size)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=CTCLoss)
    model.summary()

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        mode='min',
        verbose=1
    )
    schedule_callback = LearningRateScheduler(scheduler)

    # --- Load weights if they exist ---
    if os.path.exists(weight_local):
        print("Found local weights. Loading...")
        model.load_weights(weight_local)
        print("Weights loaded from local file.")
    else:
        try:
            print("Downloading weights from URL...")
            gdown.download(weight_url, weight_local, quiet=False)
            model.load_weights(weight_local)
            print("Weights downloaded and loaded.")
        except Exception as e:
            print("❌ Failed to download/load weights:", e)
            print("⚠️ Proceeding with random initialization.")

    # --- Train ---
    if train_params.get("training", True):
        print("\n🚀 Starting model training...")
        model.fit(
            train_pipeline,
            validation_data=test_pipeline,
            epochs=epochs,
            callbacks=[checkpoint_callback, schedule_callback]
        )
        print("✅ Training completed.")
    else:
        print("🚫 Training skipped as per configuration.")

    # --- Save final weights ---
    final_weights_path = "models/lipread_model.weights.h5"
    model.save_weights(final_weights_path)
    print(f"✅ Final model weights saved to: {final_weights_path}")

    # --- Evaluate ---
    test_loss = model.evaluate(test_pipeline)
    print(f"📊 Test loss: {test_loss:.4f}")

if __name__ == '__main__':
    main()

import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import tempfile

# Set up GPU configurations
@st.cache_resource
def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return True
        except RuntimeError as e:
            st.error(f"GPU setup error: {e}")
    return False

# Load TFLite model (cached)
@st.cache_resource
def load_tflite_model():
    try:
        # Use a relative path for the model file in the repository
        model_path = "crime_detection_model.tflite"
        # Load the TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        return None

def extract_frames(video_path, num_frames=16):
    """Extract evenly spaced frames from a video file"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to extract (evenly spaced)
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count in frame_indices:
            # Convert to RGB (from BGR) and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    
    # Ensure we have exactly num_frames
    while len(frames) < num_frames:
        # Duplicate the last frame if needed
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    return np.array(frames)

def preprocess_frame(frame):
    """Preprocess a single frame for model input using ImageNet normalization"""
    # Convert to float and scale to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # Normalize according to ImageNet stats
    frame -= np.array([0.485, 0.456, 0.406])
    frame /= np.array([0.229, 0.224, 0.225])
    
    return frame

def process_video_for_inference(video_path, num_frames=16):
    """Process a video for inference"""
    try:
        frames = extract_frames(video_path, num_frames)
        
        # Preprocess frames
        processed_frames = np.array([preprocess_frame(frame) for frame in frames])
        
        # Add batch dimension
        processed_frames = np.expand_dims(processed_frames, axis=0)
        
        return processed_frames, frames
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None, None

def predict_crime_tflite(interpreter, video_path):
    """Predict if a video contains crime using TFLite model"""
    try:
        # Process video
        frames, original_frames = process_video_for_inference(video_path)
        if frames is None:
            return "Error", 0.0, None
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Check input shape
        expected_shape = input_details[0]['shape']
        if list(frames.shape) != list(expected_shape):
            # Try to reshape if needed
            frames = np.reshape(frames, expected_shape)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], frames)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get prediction and confidence
        pred_class = np.argmax(output_data[0])
        confidence = float(output_data[0][pred_class] * 100)  # Convert to Python float
        
        prediction = "Violent" if pred_class == 1 else "Non-violent"
        
        return prediction, confidence, original_frames
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error", 0.0, None

def visualize_prediction(frames, prediction, confidence):
    """Create a visualization of the prediction"""
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    fig.suptitle(f"Prediction: {prediction} (Confidence: {confidence:.2f}%)", 
                 color='red' if prediction == 'Violent' else 'green', 
                 fontsize=16)
    
    # Display frames
    for i, ax in enumerate(axs.flat):
        if i < len(frames):
            ax.imshow(frames[i])
            ax.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(
        page_title="Crime Detection System",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("Video Crime Detection System")
    st.write("Upload a video to detect if it contains violent content.")
    
    # Sidebar with app information
    with st.sidebar:
        st.header("About")
        st.write("""
        This application analyzes video content to detect potential violent scenes using a 
        TensorFlow Lite model. Simply upload a video file to get started.
        """)
        st.write("GPU Status:", "Available" if setup_gpu() else "Not Available")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        process_button = st.button("Process Video", type="primary")
        
        if uploaded_file is not None:
            # Create a temporary file to store the video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_path = tmp_file.name
            
            # Display uploaded video
            st.video(video_path)
    
    # Process the video when the button is clicked
    if uploaded_file is not None and process_button:
        with st.spinner("Processing video..."):
            # Load TFLite model
            interpreter = load_tflite_model()
            
            if interpreter:
                # Run inference
                with col2:
                    prediction, confidence, frames = predict_crime_tflite(interpreter, video_path)
                    
                    if prediction != "Error" and frames is not None:
                        # Create result container
                        result_container = st.container()
                        
                        # Show prediction
                        if prediction == "Violent":
                            result_container.error(f"⚠️ VIOLENT CONTENT DETECTED ⚠️")
                        else:
                            result_container.success(f"✅ Non-violent content")
                        
                        # Confidence meter - ensure value is a Python float
                        result_container.progress(float(confidence/100), text=f"Confidence: {confidence:.2f}%")
                        
                        # Visualization
                        fig = visualize_prediction(frames[:8], prediction, confidence)
                        result_container.pyplot(fig)
                        
                    else:
                        st.error("Failed to process the video.")
            else:
                st.error("Failed to load model. Make sure 'crime_detection_model.tflite' is in the root directory.")
        
        # Clean up temporary file
        try:
            os.unlink(video_path)
        except:
            pass

if __name__ == "__main__":
    main()

pip install gradio roboflow supervision opencv-python pillow
import gradio as gr
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
import os
from PIL import Image
import io
import logging
import tempfile
from pathlib import Path
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define custom colors for annotations
COLORS = sv.ColorPalette.from_hex(['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff', '#ff00ff'])

class ObjectDetector:
    def __init__(self, api_key, project_name):
        try:
            self.rf = Roboflow(api_key=api_key)
            self.project = self.rf.workspace().project(project_name)
            self.model = self.project.version(1).model
            self.api_key = api_key
            self.project_name = project_name
            self.confidence_threshold = 40
            self.overlap_threshold = 30
        except Exception as e:
            logger.error(f"Error initializing ObjectDetector: {str(e)}")
            raise

    def set_thresholds(self, confidence, overlap):
        self.confidence_threshold = confidence
        self.overlap_threshold = overlap

    def process_frame(self, frame):
        try:
            # Save frame temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, frame)

            # Get predictions
            result = self.model.predict(temp_path, confidence=self.confidence_threshold, overlap=self.overlap_threshold).json()
            
            # Extract detection information
            labels = []
            xyxy = []
            confidence = []
            
            for prediction in result["predictions"]:
                labels.append(prediction["class"])
                x1 = prediction["x"] - prediction["width"] / 2
                y1 = prediction["y"] - prediction["height"] / 2
                x2 = prediction["x"] + prediction["width"] / 2
                y2 = prediction["y"] + prediction["height"] / 2
                xyxy.append([x1, y1, x2, y2])
                confidence.append(prediction["confidence"])

            # Clean up temp file
            os.unlink(temp_path)
            
            if len(xyxy) == 0:
                return frame, "No objects detected"
            
            # Create Detections object
            detections = sv.Detections(
                xyxy=np.array(xyxy, dtype=float),
                confidence=np.array(confidence),
                class_id=np.arange(len(labels))
            )
            
            # Annotate frame
            annotated_frame = frame.copy()
            box_annotator = sv.BoxAnnotator(color=COLORS, thickness=2)
            label_annotator = sv.LabelAnnotator(color=COLORS, text_thickness=2, text_scale=1)
            
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=[f"{label} ({conf:.2f})" for label, conf in zip(labels, confidence)]
            )

            return annotated_frame, labels, confidence

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, [], []

    def process_video(self, video_path, progress=None):
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create output video writer
            output_path = video_path.replace('.mp4', '_detected.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            all_detections = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                annotated_frame, labels, confidences = self.process_frame(frame)
                out.write(annotated_frame)
                
                all_detections.extend(labels)
                frame_count += 1
                
                if progress is not None:
                    progress(frame_count / total_frames)
                
            cap.release()
            out.release()
            
            return output_path, all_detections

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return None, []

def create_gradio_interface():
    # Initialize detector with error handling
    try:
        detector = ObjectDetector(
            api_key="g0J4VJVO2OUyQLWvL2Am",
            project_name="16-10-2022-1-kgxxo"
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {str(e)}")
        detector = None

    def process_image(image, conf_threshold, overlap_threshold):
        try:
            if detector is None:
                return None, "Detector not initialized"
            
            detector.set_thresholds(conf_threshold, overlap_threshold)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            annotated_frame, labels, confidences = detector.process_frame(frame)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            summary = f"Found {len(labels)} objects:\n" + "\n".join(
                [f"- {label} ({conf:.2f})" for label, conf in zip(labels, confidences)]
            )
            
            return annotated_frame, summary
        except Exception as e:
            return None, f"Error: {str(e)}"

    def process_video_file(video_path, conf_threshold, overlap_threshold, progress=gr.Progress()):
        try:
            if detector is None:
                return None, "Detector not initialized"
            
            detector.set_thresholds(conf_threshold, overlap_threshold)
            output_path, all_detections = detector.process_video(video_path, progress)
            
            if output_path is None:
                return None, "Error processing video"
            
            detection_counts = {}
            for label in all_detections:
                detection_counts[label] = detection_counts.get(label, 0) + 1
            
            summary = "Video Processing Complete\n\nDetection Summary:\n" + "\n".join(
                [f"- {label}: {count} instances" for label, count in detection_counts.items()]
            )
            
            return output_path, summary
        except Exception as e:
            return None, f"Error: {str(e)}"

    def process_webcam(image, conf_threshold, overlap_threshold):
        return process_image(image, conf_threshold, overlap_threshold)

    # Create modern interface with tabs
    with gr.Blocks(theme=gr.themes.Glass()) as iface:
        gr.Markdown("# 🔍 Advanced Object Detection System")
        
        with gr.Tabs():
            with gr.Tab("📷 Image Detection"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Upload Image")
                        with gr.Row():
                            conf_slider = gr.Slider(minimum=0, maximum=100, value=40, label="Confidence Threshold")
                            overlap_slider = gr.Slider(minimum=0, maximum=100, value=30, label="Overlap Threshold")
                        detect_btn = gr.Button("Detect Objects", variant="primary")
                    with gr.Column():
                        image_output = gr.Image(label="Detected Objects")
                        text_output = gr.Textbox(label="Detection Results", lines=5)
                
                detect_btn.click(
                    process_image,
                    inputs=[image_input, conf_slider, overlap_slider],
                    outputs=[image_output, text_output]
                )

            with gr.Tab("🎥 Video Detection"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Video")
                        with gr.Row():
                            video_conf_slider = gr.Slider(minimum=0, maximum=100, value=40, label="Confidence Threshold")
                            video_overlap_slider = gr.Slider(minimum=0, maximum=100, value=30, label="Overlap Threshold")
                        video_detect_btn = gr.Button("Process Video", variant="primary")
                    with gr.Column():
                        video_output = gr.Video(label="Processed Video")
                        video_text_output = gr.Textbox(label="Detection Summary", lines=5)
                
                video_detect_btn.click(
                    process_video_file,
                    inputs=[video_input, video_conf_slider, video_overlap_slider],
                    outputs=[video_output, video_text_output]
                )

        
    


        gr.Markdown("""
        ### Features
        - Support for image, video, and webcam input
        - Adjustable confidence and overlap thresholds
        - Real-time detection visualization
        - Detection summary and statistics
        - Modern glass-themed UI
        """)

    return iface

if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch(share=True) 
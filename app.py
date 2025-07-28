import gradio as gr
from PIL import Image
import os
import torch
import mediapipe as mp
from transformers import SamModel, SamProcessor
from diffusers.utils import load_image
from torchvision import transforms

# Load model once
model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50")
processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")

def get_shoulder_coordinates(image: Image.Image):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        height, width, _ = image_rgb.shape
        landmarks = results.pose_landmarks.landmark
        left = (int(landmarks[11].x * width), int(landmarks[11].y * height))
        right = (int(landmarks[12].x * width), int(landmarks[12].y * height))
        return left, right
    return None

def try_on(person_img, tshirt_img):
    coordinates = get_shoulder_coordinates(person_img)
    if coordinates is None:
        return "No pose detected", None
    
    left_shoulder, right_shoulder = coordinates
    input_points = [[[left_shoulder[0], left_shoulder[1]], [right_shoulder[0], right_shoulder[1]]]]

    inputs = processor(person_img, input_points=input_points, return_tensors="pt")
    outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),
                                                          inputs["original_sizes"].cpu(),
                                                          inputs["reshaped_input_sizes"].cpu())

    mask_tensor = masks[0][0][2].to(dtype=torch.uint8)
    mask = transforms.ToPILImage()(mask_tensor * 255)

    tshirt_img = tshirt_img.resize(person_img.size, Image.LANCZOS)
    result = Image.composite(tshirt_img, person_img, mask)
    return result

demo = gr.Interface(fn=try_on, 
                    inputs=["image", "image"], 
                    outputs="image",
                    title="Virtual Try-On using SlimSAM",
                    description="Upload a person image and a t-shirt image.")

demo.launch()

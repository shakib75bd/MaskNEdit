import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SamModel, SamProcessor

def mask_and_retouch_face(image_path, output_path):
    # Load SAM model and processor
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use face detection (OpenCV's built-in face detector)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print("No faces detected in the image.")
        return

    # For each detected face, generate a mask using SAM
    for (x, y, w, h) in faces:
        # Create input point for SAM (center of face)
        input_points = [[[x + w//2, y + h//2]]]

        # Process image and get SAM prediction
        inputs = processor(image, input_points=input_points, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Get mask
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        mask = masks[0][0][0].numpy()

        # Apply retouching (simple example: skin smoothing)
        # Convert to uint8 mask for OpenCV
        binary_mask = (mask > 0).astype(np.uint8) * 255

        # Apply slight Gaussian blur for simple skin smoothing
        blurred = cv2.GaussianBlur(image, (21, 21), 11)

        # Combine original and blurred images using the mask
        mask_3ch = np.stack([binary_mask/255.0]*3, axis=2)
        retouched = image * (1 - mask_3ch) + blurred * mask_3ch

        # Save result
        retouched_img = Image.fromarray(retouched.astype(np.uint8))
        retouched_img.save(output_path)

        # For visualization
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")

        plt.subplot(1, 3, 2)
        plt.imshow(binary_mask, cmap='gray')
        plt.title("Face Mask")

        plt.subplot(1, 3, 3)
        plt.imshow(retouched.astype(np.uint8))
        plt.title("Retouched Image")
        plt.savefig(output_path.replace('.jpg', '_comparison.jpg'))

        print(f"Processed image saved to {output_path}")
        break  # Process only the first face for this example

if __name__ == "__main__":
    input_image = "./sssn.jpg"
    output_image = "./retouched_image.jpg"
    mask_and_retouch_face(input_image, output_image)

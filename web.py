import streamlit as st
from huggingface_hub import hf_hub_download
from huggingface_hub import login
from PIL import Image
import os
# import cv2
import numpy as np
import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2

# Defining image classes and sizes
CLASSES = ["background", "car", "wheel", "lights", "window"]
INFER_WIDTH = 256
INFER_HEIGHT = 256

# Normalization statistics for ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Defining a computing device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Authentication for access to the repository on hugging face
login(token="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")

# Checking the file availability best_unet_model.pt in the models directory
if not os.path.exists('models/best_unet_model.pt'):
    # Loading the JIT model from repository on hugging face
    hf_hub_download(
        repo_id='melsmm/car-segmentation-unet',
        filename='best_unet_model.pt',
        local_dir='models'
    )

# Loading the JIT model
best_model = torch.jit.load('models/best_unet_model.pt', map_location=DEVICE)

def get_validation_augmentation():
    """Get augmentations for validation."""
    test_transform = [
        albu.LongestMaxSize(max_size=INFER_HEIGHT, always_apply=True),
        albu.PadIfNeeded(min_height=INFER_HEIGHT, min_width=INFER_WIDTH, always_apply=True),
        albu.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return albu.Compose(test_transform)

def infer_image(image):
    """To get a mask on an image using the Unet model."""
    original_height, original_width, _ = image.shape

    # Application of augmentations
    augmentation = get_validation_augmentation()
    augmented = augmentation(image=image)
    image_transformed = augmented['image']

    # # # # Converting an image to a Pytorch tensor and moving it to a device
    x_tensor = torch.from_numpy(image_transformed).to(DEVICE).unsqueeze(0).permute(0, 3, 1, 2).float()

    # Inference
    best_model.eval()
    with torch.no_grad():
        pr_mask = best_model(x_tensor)

    # Converting the output to a numpy array and removing the package dimension
    pr_mask = pr_mask.squeeze().cpu().detach().numpy()

    # Getting the class with the highest probability for each pixel
    label_mask = np.argmax(pr_mask, axis=0)

    # Determining the number of pixels that will appear on the sides of the paddings and cropping them
    if original_height > original_width:
        delta_pixels = int(((original_height-original_width)/2)/original_height * INFER_HEIGHT)
        image_cropped = image_transformed[:, delta_pixels + 1: INFER_WIDTH - delta_pixels - 1]
        mask_cropped = label_mask[:, delta_pixels + 1 : INFER_WIDTH - delta_pixels - 1]
    elif original_height < original_width:
        delta_pixels = int(((original_width-original_height)/2)/original_width * INFER_WIDTH)
        image_cropped = image_transformed[delta_pixels + 1: INFER_HEIGHT - delta_pixels - 1, :]
        mask_cropped = label_mask[delta_pixels + 1: INFER_HEIGHT - delta_pixels - 1, :]
    else:
        mask_cropped = label_mask
        image_cropped = image_transformed

    # Resizing the mask back to the original image size
    label_mask_real_size = cv2.resize(
        mask_cropped, (original_width, original_height), interpolation=cv2.INTER_NEAREST
    )

    return label_mask_real_size

def adjust_hsv(image, mask, h_adjust, s_adjust, v_adjust, index):
    """Adjusting the HSV value in the image in the area where mask == index."""
    # Converting an image to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(image_hsv)
    
    # Applying adjustments only to the area where mask == index
    h[mask == index] = np.clip(h[mask == index] + h_adjust, 0, 179)
    s[mask == index] = np.clip(s[mask == index] + s_adjust, 0, 255)
    v[mask == index] = np.clip(v[mask == index] + v_adjust, 0, 255)
    
    # Combining HSV channels back into a single image
    image_hsv_adjusted = cv2.merge([h, s, v])
    
    # Convert the image back to RGB for display
    image_rgb_adjusted = cv2.cvtColor(image_hsv_adjusted.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return image_rgb_adjusted

def display_image(image):
    """Image Display."""
    st.image(image, use_column_width=True)

def upload_image(label):
    """Uploading an image."""
    uploaded_file = st.file_uploader(label, type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image_data = np.array(Image.open(uploaded_file))
        return image_data
    return None

def main():
    st.set_page_config(
        page_title="Image Processor",
        page_icon='🎨',
        layout="wide",
        initial_sidebar_state="expanded",)

    st.title('Image Correction Tool')

    # Uploading an image
    image = upload_image('Upload an image')

    # Checking that the image has been uploaded
    if image is not None:
        # Selecting values to adjust HSV
        h_adjust = st.sidebar.slider('Adjusting the hue (H) (-179 до 179)', -179, 179, 0)
        s_adjust = st.sidebar.slider('Adjusting the saturation (S) (-255 до 255)', -255, 255, 0)
        v_adjust = st.sidebar.slider('Adjusting the value (V) (-255 до 255)', -255, 255, 0)

        # Selecting the value to change in the mask using the drop-down list
        mask_value = st.sidebar.selectbox('Select an area of interest', CLASSES)

        # We are looking for the index of the value in the list
        index = CLASSES.index(mask_value)

        mask = infer_image(image)

        # Applying HSV adjustments
        adjusted_image = adjust_hsv(image, mask, h_adjust, s_adjust, v_adjust, index)

        # Displaying the original image and the adjusted image in two columns
        col1, col2, _ = st.columns(3)
        with col1:
            display_image(image)
        with col2:
            display_image(adjusted_image)


if __name__ == '__main__':
    main()

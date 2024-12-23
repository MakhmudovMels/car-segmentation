import streamlit as st
from PIL import Image
import cv2
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

# Loading the JIT model
best_model = torch.jit.load('models/best_unet_model.pt', map_location=DEVICE)

def get_validation_augmentation():
    """Получить аугментации для валидации."""
    test_transform = [
        albu.LongestMaxSize(max_size=INFER_HEIGHT, always_apply=True),
        albu.PadIfNeeded(min_height=INFER_HEIGHT, min_width=INFER_WIDTH, always_apply=True),
        albu.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return albu.Compose(test_transform)

def infer_image(image):
    """Получить маску на изображении с помощью модели Unet."""
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
    """Корректировка значения HSV на изображении в области, где mask == index."""
    # Преобразование изображения в HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(image_hsv)
    
    # Применение корректировок только к области, где mask == index
    h[mask == index] = np.clip(h[mask == index] + h_adjust, 0, 179)
    s[mask == index] = np.clip(s[mask == index] + s_adjust, 0, 255)
    v[mask == index] = np.clip(v[mask == index] + v_adjust, 0, 255)
    
    # Объединение каналов HSV обратно в одно изображение
    image_hsv_adjusted = cv2.merge([h, s, v])
    
    # Преобразование изображения обратно в RGB для отображения
    image_rgb_adjusted = cv2.cvtColor(image_hsv_adjusted.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return image_rgb_adjusted

def display_image(image):
    """Отображение изображения."""
    st.image(image, use_column_width=True)

def upload_image(label):
    """Загрузка изображения."""
    uploaded_file = st.file_uploader(label, type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image_data = np.array(Image.open(uploaded_file))
        return image_data
    return None

def main():
    st.set_page_config(
        page_title="Обрабочик изображений",
        page_icon='🎨',
        layout="wide",
        initial_sidebar_state="expanded",)

    st.title('Инструмент корректировки изображений')

    # Загрузка изображения
    image = upload_image('Загрузите изображение')

    # Проверка, что изображение загружено
    if image is not None:
        # Выбор значений для корректировки HSV
        h_adjust = st.sidebar.slider('Корректировка оттенка (H) (-179 до 179)', -179, 179, 0)
        s_adjust = st.sidebar.slider('Корректировка насыщенности (S) (-255 до 255)', -255, 255, 0)
        v_adjust = st.sidebar.slider('Корректировка освещения (V) (-255 до 255)', -255, 255, 0)

        # Выбор значения для изменения в маске с помощью выпадающего списка
        mask_value = st.sidebar.selectbox('Выберите интересующую область', CLASSES)

        # Ищем индекс значения в списке
        index = CLASSES.index(mask_value)

        mask = infer_image(image)

        # Применение корректировок HSV
        adjusted_image = adjust_hsv(image, mask, h_adjust, s_adjust, v_adjust, index)

        # Отображение исходного изображения и скорректированного изображения в двух столбцах
        col1, col2, _ = st.columns(3)
        with col1:
            display_image(image)
        with col2:
            display_image(adjusted_image)


if __name__ == '__main__':
    main()

import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from model import EfficientNet, KappaOptimizer
import torch.optim as optim
import cv2
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
import torch
import fastai
# from fastai.vision import open_image
# from fastai.callbacks import SaveModelCallback
# from fastai.basic_train import load_learner

model_path = 'bestmodel.pth' 
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

model_state_dict = checkpoint['model']
model = EfficientNet.from_pretrained('efficientnet-b0') 

model.eval() 

data = { 0: "No DR",
         1: "Mild",
         2: "Moderate",
         3: "Proliferative DR",
         4:  "Severe"}



def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

import random

def random_rotation(image):
    angle = random.randint(-30,30)
    return transforms.functional.rotate(image,angle)

my_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.Lambda(random_rotation),
    transforms.ToPILImage(),
    transforms.ToTensor(),
#     transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
])

st.set_page_config(
    layout="wide"
)

st.title("Diabetes Retinopathy Detection")
coefficients = [0.518652, 1.496484, 2.554199, 3.543066]

uploaded_file = st.file_uploader("Upload Photo", 
                                  type=["jpg", "jpeg", "png"], key="file_uploader")

if uploaded_file is not None:
    # if len(uploaded_files) > 5:
        # st.warning("Please upload up to 5 photos.")
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Decode the image data
    image = cv2.imdecode(file_bytes, 1)
    # Display the uploaded image
    # image = open_image(uploaded_file.name)
    st.image(image,caption = "Uploaded Image", channels="BGR",width = 100)

    # image = circle_crop(uploaded_file.name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (224, 224))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)
    st.image(image, caption='Processed Image', width=100)
    
    # Preprocess the images
    transformed_image = my_transform(image)

    input_tensor = transformed_image.unsqueeze(0)
    # pred_probs, _ = learn.get_preds(ds_type=DatasetType.Single, dl=[(image.data, torch.tensor([0]))])
    # pred_probs = pred_probs.squeeze().tolist()

    # weighted_pred_probs = [p * c for p, c in zip(pred_probs, coefficients)]

    # Get the predicted class index
    # predicted_class_index = weighted_pred_probs.index(max(weighted_pred_probs))

    with torch.no_grad():
        output = model(input_tensor)
        
    pred_probs = torch.nn.functional.softmax(output, dim=1).squeeze().tolist()
    weighted_pred_probs = [p * c for p, c in zip(pred_probs, coefficients)]

    predicted_class_index = weighted_pred_probs.index(max(weighted_pred_probs))

        

    st.markdown(
        """
        <style>
        .stButton>button {
        width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
        
    if st.button("Predict"):
        st.write(f'Predicted Class: {data[predicted_class_index]} ({predicted_class_index})')
        


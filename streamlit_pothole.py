#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
pipreqs r"D:\Projects\streamlit_pothole.py" --force

st.title("🕳️ Pothole Detection with YOLOv8")

# Load model from local weights folder
model_path = os.path.join("weights", r"C:\Users\Nodir\anaconda3\envs\detection\runs\detect\train11\weights\best.pt")
model = YOLO(model_path)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_column_width=True)

    # Save uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        img.save(tmp.name)
        results = model.predict(source=tmp.name, imgsz=640, conf=0.25, save=False)

        # Visualize results directly in memory
        result_img = results[0].plot()
        st.image(result_img, caption="Detected Potholes", use_column_width=True)

        # Optional: Show raw prediction data
        st.write("Detections:", results[0].boxes.data.cpu().numpy())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





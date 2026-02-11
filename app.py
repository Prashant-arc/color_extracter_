import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans


def create_color_palette(dominant_colors, palette_size=(300, 50)):
    palette = Image.new('RGB', palette_size)
    draw = ImageDraw.Draw(palette)

    swatch_width = palette_size[0] // len(dominant_colors)

    for i, color in enumerate(dominant_colors):
        draw.rectangle(
            [i * swatch_width, 0, (i + 1) * swatch_width, palette_size[1]],
            fill=tuple(color)
        )
    
    return palette 

st.title("Dominant Color Extractor")
st.write("Upload an image to extract its top 3 colors using K-Means Clustering.")


uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # open image 
    image = Image.open(uploaded_file)
    
    img_small = image.resize((150, 150))
    
    img_array = np.array(img_small)
    
    
    pixel_data = img_array.reshape(-1, 3) 

    st.write(pixel_data.shape)

    
    st.write("Computing colors...")
    
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(pixel_data)
    
    dominant_colors = kmeans.cluster_centers_.astype(int)

   
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Dominant Palette")
        # Use your function to draw the palette
        palette_img = create_color_palette(dominant_colors)
        st.image(palette_img, use_container_width=True)
        
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import random
import time
from huggingface_hub import hf_hub_download


# Load model from Hugging Face Model Hub
model_path = hf_hub_download(repo_id="koulsahil/LandCoverClassification_EuroSat", filename="eurosat_rgb_model.h5")
model = tf.keras.models.load_model(model_path)
model.eval()



# Define the class labels (replace with your EuroSAT classes)
class_labels = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

# Function to preprocess the image for the model
def preprocess_image(image):
    # Convert image to RGB if it has an alpha channel (4 channels)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize((64, 64))  # Resize to match EuroSAT input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence




# Set the page title and favicon (emoji as the icon)
st.set_page_config(
    page_title="Land Cover Classification",  # Title of the app
    page_icon="🌍",  # Use the world emoji as the icon
)





# Custom CSS for styling
st.markdown("""
<style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .stMarkdown h1 {
        color: #4CAF50;
    }
    .stMarkdown h2 {
        color: #2E86C1;
    }
    .upload-section {
        height: 150px; /* Increase the height of the upload section */
    }
    .thumbnail {
        cursor: pointer;
        border: 2px solid transparent;
        transition: border 0.3s ease;
        margin: 5px;
    }
    .thumbnail:hover {
        border: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for project information
import streamlit as st

with st.sidebar:
    st.title("About the Project 🌍")
    st.write("""
    Hi, I’m **Sahil!** I built this web app to classify satellite images using Neural Networks,
    This app leverages a **custom VGG16 deep learning model** trained on the **EuroSAT dataset** to classify land cover types in satellite images. 
    The EuroSAT dataset is a collection of **27,000 labeled satellite images** covering **10 distinct land cover classes**, such as forests, crops, industrial areas, and water bodies. 


    ### How to Use:
    1. **Select an Image**:
       - Drag and drop one of the **sample image thumbnails** to the upload section to quickly test the model.
       - Or use the dropdown menu to choose from a list of sample images, or click the select random button to choose a random image from the list.
       - Alternatively, **upload your own satellite image** by dragging and dropping it into the upload section.
    2. **Make a Prediction**:
       - Once an image is selected or uploaded, the model will automatically analyze it and display the **predicted land cover type** along with a **confidence score**.
    3. **Interpret the Results**:
       - The app will show the **top predicted class** and its confidence level.
       - You can also view the **top 3 predictions** to understand the model's certainty across multiple classes.


    ### GitHub Repository:
    Explore the code, dataset, and model training process on GitHub:  
    [GitHub Repository](https://koulmesahil.github.io/) | [LinkedIn](https://www.linkedin.com/in/sahilkoul123/)
    """)

# Main layout
st.title("Land Cover Classification from Satellite Images ")
st.write("🖼️📸 Drag and drop one of the thumbnails below, select a random image from the dropdown, or upload your own image to classify its land cover type.🌍")

# Define sample images
sample_images = {
    "Highway": "sample_images/Highway_1004.jpg",
    "Annual Crop": "sample_images/AnnualCrop_102.jpg",
    "Forest": "sample_images/Forest_1019.jpg",
    "Herbaceous Vegetation": "sample_images/HerbaceousVegetation_1024.jpg",
    "Industrial": "sample_images/Industrial_1015.jpg",
    "Pasture": "sample_images/Pasture_1023.jpg",
    "Sea Lake": "sample_images/SeaLake_1017.jpg",
    "River": "sample_images/River_1014.jpg",
    "Permenant Crop": "sample_images/PermanentCrop_1004.jpg",
    "Residential": "sample_images/Residential_1019.jpg",
    # Add more sample images here
}

# Thumbnail Section
st.write("### Sample Image Thumbnails")
cols = st.columns(len(sample_images))  # Create columns for thumbnails
for idx, (label, image_path) in enumerate(sample_images.items()):
    with cols[idx]:
        # Display the thumbnail
        image = Image.open(image_path)
        image.thumbnail((100, 100))  # Resize the image to a smaller thumbnail

        st.image(image, use_container_width=True, caption=None)


# Dropdown menu for sample images
#st.write("### Select an Image from Dropdown")
selected_image_label = st.selectbox("Choose a category to view a random image:", ["Pick a category"] + list(sample_images.keys()))

if selected_image_label != "Pick a category":
    image_path = sample_images[selected_image_label]
    with open(image_path, "rb") as file:
        uploaded_file = file.read()
    st.session_state.uploaded_file = uploaded_file
    st.session_state.selected_image_label = selected_image_label


# Add a "Select Random" button
if st.button("Random Selection"):
    random_label, random_image_path = random.choice(list(sample_images.items()))
    with open(random_image_path, "rb") as file:
        uploaded_file = file.read()
    st.session_state.uploaded_file = uploaded_file
    st.session_state.selected_image_label = random_label
    st.success(f"Randomly selected image: **{random_label}**")


# JavaScript & CSS to dynamically change the border color
st.markdown(
    """
    <style>
        /* Base styling for the upload box */
        div.stFileUploader {
            height: 250px !important;
            width: 100% !important;
            border: 3px dashed grey !important;
            padding: 40px !important;
            border-radius: 10px;
            transition: all 0.3s ease-in-out;
        }

        /* Upload text styling */
        div.stFileUploader > label {
            font-size: 20px !important;
            font-weight: bold !important;
            color: grey !important;
            text-align: center !important;
        }

        /* JavaScript to change colors dynamically */
        <script>
            function updateUploaderColor() {
                var uploader = document.querySelector("div.stFileUploader");
                if (uploader && uploader.querySelector("input").files.length > 0) {
                    uploader.style.borderColor = "#4CAF50";
                    uploader.style.color = "#4CAF50";
                } else {
                    uploader.style.borderColor = "grey";
                    uploader.style.color = "grey";
                }
            }
            
            document.addEventListener("DOMContentLoaded", function() {
                var fileInput = document.querySelector("div.stFileUploader input");
                if (fileInput) {
                    fileInput.addEventListener("change", updateUploaderColor);
                }
            });
        </script>
    </style>
    """,
    unsafe_allow_html=True,
)

st.write("### Upload Your Image")
uploaded_file = st.file_uploader(
    "Drag and drop an image here",
    type=["jpg", "jpeg", "png"],
    key="uploader",
    accept_multiple_files=False,
    help="Upload an image to classify its land cover type."
)

if uploaded_file:
    st.markdown(
        """
        <style>
            div.stFileUploader {
                border-color: #4CAF50 !important;
                color: #4CAF50 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Display the uploaded/selected image and make predictions
if uploaded_file is not None:
    try:
        file_type = uploaded_file.type
        if file_type not in ["image/jpeg", "image/png"]:
            st.error("Unsupported file type. Please upload a JPG or PNG image.")
        else:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_file = uploaded_file.getvalue()
            st.session_state.selected_image_label = "Uploaded Image"
    except Exception as e:
        st.error(f"Error loading image: {e}")


# Display the uploaded/selected image and make predictions
if "uploaded_file" in st.session_state:
    image = Image.open(io.BytesIO(st.session_state.uploaded_file))

    # Display the image with a smaller size
    st.image(image, caption=f"Selected Image: {st.session_state.selected_image_label}", width=300)
    
    # Show balloons effect after the prediction is done
    with st.spinner("Analyzing the image..."):
        time.sleep(1)  # Simulate processing delay
        predicted_class, confidence = predict(image)
        st.balloons()  # Display balloons when prediction is complete

    # Display the prediction results in a more prominent section
    st.markdown("## Prediction Results")
    st.success(f"**Predicted Class:** {predicted_class}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")
    
    # Visualize confidence as a progress bar with a label
    st.markdown("**Confidence Level:**")
    st.progress(float(confidence))
    
    # Show top 3 predictions
    st.markdown("### Top 3 Predictions")
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    top_indices = np.argsort(predictions[0])[-3:][::-1]  # Get top 3 predictions
    for i in top_indices:
        st.write(f"- **{class_labels[i]}**: {predictions[0][i] * 100:.2f}%")



# Footer
st.markdown("---")
st.markdown("""
[GitHub Repository](https://koulmesahil.github.io/) | [LinkedIn](https://www.linkedin.com/in/sahilkoul123/)
""")
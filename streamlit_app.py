import streamlit as st
import pandas as pd
import numpy as np

from roboflow import Roboflow
import cv2


# Load the trained model from Roboflow
rf = Roboflow(api_key="5VImoUONxH3MRjhhlGls")
project = rf.workspace().project("mlops-final-project-object-detection")
model = project.version(1).model

def detect_objects(img, conf, overlap):
    return model.predict(img, confidence=conf, overlap=overlap).json()
    
def calculate_amount(pred_json):
    df_count = pd.DataFrame(pred_json['predictions'])
    df_count['amount_php'] = df_count['class'].str.split('P').str.get(-1).astype(float)
    return (df_count.groupby('class').agg(Quantity=('amount_php', 'size'),
                                         Total=('amount_php', 'sum'))
                    .reset_index().rename(columns={'class': 'Value'}))
    
# Define a function to get predictions and annotate the image
def get_predictions(img, pred_json):
    
    # Create a copy of the input image
    annotated_img = img.copy()
    
    # Annotate the image with predictions
    for bounding_box in pred_json['predictions']:
        pred_value = bounding_box['class']
        pred_conf = bounding_box['confidence']

        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2

        bbox_start, bbox_end = (int(x0), int(y0)), (int(x1), int(y1))
        
        # Calculate the size of the text rectangle based on text size
        text = f"{pred_value} | Confidence: {pred_conf:.1%}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 2)
        
        text_end = (int(x0) + text_width + 10, int(y0) - text_height - 10)
        
        cv2.rectangle(annotated_img, bbox_start, bbox_end, color=(206, 255, 0), thickness=2)
        cv2.rectangle(annotated_img, bbox_start, text_end, color=(206, 255, 0), thickness=cv2.FILLED)
        
        cv2.putText(
            annotated_img,
            text,
            (int(x0), int(y0) - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=(0, 0, 0)
        )
        
    return annotated_img



# Main Streamlit app
def main():

    # Set the page width
    st.set_page_config(layout="wide")

    st.markdown("<h1 style='text-align: center;'>MoneyGuard: Secure Banknote Scanner</h1>", unsafe_allow_html=True)
    
    # Define columns for layout
    col1, col2, col3 = st.columns(3, gap='medium')

    # Upload an image in the left column
    with col1:
        st.subheader('Upload an image')
        uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"])
        
    if uploaded_image is not None:
        
        # Sliders for confidence and overlap in the left column
        with col1:
            conf = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
            overlap = st.slider("Overlap Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
            
            if st.button("Detect Objects"):
            
                # Read the uploaded image with OpenCV
                image_bytes = uploaded_image.read()
                img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                # Display a message
                message_placeholder = st.empty()
                message_placeholder.text("Detecting objects...")
                
                # Detect objects in the image
                pred_json = detect_objects(img, conf, overlap)
                
                # Get predictions and annotate the image
                annotated_img = get_predictions(img, pred_json)
                
                message_placeholder.empty()    

                # Center column for displaying the annotated image
                with col2:
                    st.subheader('Detected banknotes')
                    # Display the annotated image
                    st.image(annotated_img, use_column_width=True)

                # Right column for displaying the summary table
                with col3:
                    st.subheader('Amount in PHP')
                  
                    if 'pred_json' in locals():
                        df_amount = calculate_amount(pred_json)
                        total_value = df_amount.Total.sum()
                        
                        # Display total value
                        st.markdown("<div style='background-color: #CDFF00; color: black; padding: 5px; text-align: center;'>" ##00FFCE
                                    f"<p style='font-size: 24px; font-weight: bold; margin-top: 0; margin-bottom: 0;'>{total_value}</p>"
                                    "<p style='font-style: italic; font-size: small; margin-top: 0; margin-bottom: 0;'>Total Amount</p>"
                                    "</div>", unsafe_allow_html=True)
                        
                        # Display the summary
                        st.dataframe(df_amount, use_container_width=True, hide_index=True)
                    

if __name__ == "__main__":
    main()
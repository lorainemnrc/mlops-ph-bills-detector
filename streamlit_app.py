import streamlit as st
import pandas as pd
import numpy as np
import os, io

from roboflow import Roboflow
from gtts import gTTS
import inflect
import cv2


# Load the trained model from Roboflow
rf = Roboflow(api_key="5VImoUONxH3MRjhhlGls")
project = rf.workspace().project("mlops-final-project-object-detection")
model = project.version(3).model

# Define global parameters
language = 'en'
file_path = os.path.dirname(__file__)
banner_img = os.path.join(file_path, "mg_logo.png")

def detect_objects(img, conf, overlap):
    """
    Return detected objects in an image using a pre-trained model.
    
    Parameters:
    -------------------
    img : str or bytes
        The input image to be processed. It can be a file path (str) or image data (bytes).

    conf : float
        The confidence threshold for object detection. Detected objects with a confidence
        score below this threshold will not be included in the results.

    overlap : float
        The overlap threshold for non-maximum suppression (NMS). It controls the degree
        of overlap allowed between bounding boxes. Lower values result in stricter NMS.

    Returns:
    dict
        A dictionary containing the detected objects and their associated information.
        The structure of the dictionary depends on the model and may include information
        such as object labels, bounding box coordinates, and confidence scores.
        
    Note:
    - The `model` used for object detection should be pre-trained and compatible with
      the `predict` method, accepting parameters for confidence and overlap thresholds.
    - Ensure that the necessary dependencies and model files are available and properly
      configured for object detection.
    """
    return model.predict(img, confidence=conf, overlap=overlap).json()
    
def calculate_amount(pred_json):
    """
    Return calculated values such as quantity and total amount
    for different classes of predictions as a pandas DataFrame.    

    Parameters:
    -------------------
    pred_json : dict
        A JSON object containing prediction data. It is expected to have a structure
        similar to {'predictions': [{'class': 'PredictionClass1'}, {'class': 'PredictionClass2'}, ...]}.

    Returns:
    df_count : pandas DataFrame
        A DataFrame summarizing the quantity and total amount for each prediction class.
        The DataFrame contains columns for 'Value' (class name), 'Quantity' (number of occurrences),
        'Total' (sum of amounts), 'qty_words' (quantity in words), and 'value_num' (numeric value).

    Note:
    - The input JSON should have a 'predictions' key containing a list of dictionaries.
    - The 'amount_php' field in the input JSON is expected to be in Philippine Peso format (e.g., 'P100.50').
    - The function uses the 'number_to_words' function to convert quantity to words.
    """
    df_count = pd.DataFrame(pred_json['predictions'])
    df_count['amount_php'] = df_count['class'].str.split('P').str.get(-1).astype(float)
    df_count = (df_count.groupby('class').agg(Quantity=('amount_php', 'size'),
                                         Total=('amount_php', 'sum'))
                        .reset_index().rename(columns={'class': 'Value'}))
    df_count['qty_words'] = df_count['Quantity'].apply(number_to_words)
    df_count['value_num'] = df_count['Value'].str.strip('P').astype(float)
    return df_count.sort_values('value_num', ascending=True)
                    
# Define a function to format each row
def format_row(row):
    """
    Format a row of data into a human-readable string.

    Parameters:
    -------------------
    row : pandas Series or dict
        A row of data containing at least 'qty_words' and 'Value' (class name)
        information.

    Returns:
    formatted_str : str
        A formatted string representing the quantity and value in pesos, e.g.,
        "One hundred pesos."
        
    Note:
    - The input row should contain at least the 'qty_words' and 'Value' fields.
    - The 'Value' field is expected to be in Philippine Peso format (e.g., 'P100.00').
    - The 'qty_words' field should contain the quantity description in words.
    """
    return f"{row['qty_words']} {row['Value'].strip('P')} pesos"

# Create a function to convert numeric values to words
def number_to_words(number):
    """
    Convert a numeric value into words.

    Parameters:
    -------------------
    number : int or float
        The numeric value to be converted into words.

    Returns:
    words : str
        A string representing the numeric value in words.

    Note:
    - The function uses the `inflect` library to perform the conversion.
    - Ensure that the `inflect` library is properly installed and imported to use this function.
    """
    p = inflect.engine()
    return p.number_to_words(number)
   
# Generate audio from text   
def get_audio(df, total_value):
    """
    Generate audio from text using a text-to-speech model.

    Parameters:
    -------------------
    df : pandas DataFrame
        A DataFrame containing formatted text rows.

    total_value : str or numeric
        The total value to be included in the audio.

    language : str, optional (default='en')
        The language for text-to-speech synthesis. Use ISO language codes (e.g., 'en' for English).

    Returns:
    audio_bytes : io.BytesIO
        An audio representation of the formatted text and total value in BytesIO format.

    Note:
    - The function uses the gTTS (Google Text-to-Speech) library to generate audio.
    - The `language` parameter specifies the language for the text-to-speech synthesis.
    """
    # Read the results using text-to-speech models
    formatted_strings = df.apply(format_row, axis=1).tolist()
    start_text = "Here are the bills detected: "
    end_text = f" The total amount is: {total_value}"

    # Check if there is more than one row
    if len(formatted_strings) > 1:
        formatted_strings[-1] = "and " + formatted_strings[-1]
        
    # Join the formatted strings into a single string
    text_prompt = start_text + ", ".join(formatted_strings) + end_text
    
    # Create a gTTS object with the text
    tts = gTTS(text=text_prompt, lang=language, slow=False)

    # Save the audio as a BytesIO object
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    
    return audio_bytes
    
# Define a function to get predictions and annotate the image
def get_predictions(img, pred_json):
    """
    Return the annotate an image with object detection predictions.

    Parameters:
    -------------------
    img : numpy.ndarray
        The input image on which predictions will be annotated. It should be a NumPy array
        representing an image in standard format.

    pred_json : dict
        A JSON object containing prediction data. It is expected to have a structure similar to:
        {'predictions': [{'class': 'ObjectClass1', 'confidence': 0.85, 'x': 100, 'y': 200, 'width': 50, 'height': 50},
                         {'class': 'ObjectClass2', 'confidence': 0.92, 'x': 150, 'y': 250, 'width': 30, 'height': 30},
                         ...]}

    Returns:
    annotated_img : numpy.ndarray
        Annotated image with bounding boxes and prediction labels.

    Note:
    - The function uses OpenCV (cv2) for image processing and annotation.
    - The input image should be in NumPy array format.
    - The prediction data is expected to include class labels, confidence scores, and bounding box coordinates.
    """
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
        text = f"{pred_value} | {pred_conf:.1%}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 2)
        
        text_end = (int(x0) + text_width + 10, int(y0) - text_height - 10)
        
        cv2.rectangle(annotated_img, bbox_start, bbox_end, color=(142, 0, 70), thickness=2)
        cv2.rectangle(annotated_img, bbox_start, text_end, color=(142, 0, 70), thickness=cv2.FILLED) #206, 255, 0; 0, 255, 206
        
        cv2.putText(
            annotated_img,
            text,
            (int(x0), int(y0) - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=(255, 255, 255)
        )
        
    return annotated_img

# Main Streamlit app
def main():

    # Set the page width
    st.set_page_config(layout="wide")
    
    # Set title
    custom_color = "#D10A84"  # Replace with your desired color code
    
    # Apply custom CSS style for the title text color
    st.write(f'<style>h1{{color: {custom_color};}}</style>', unsafe_allow_html=True)
    st.title('Money Guard: Making sure the price is right!')

    
    st.divider()
    # st.markdown("<h1 style='text-align: center;'>MoneyGuard: Secure Banknote Scanner</h1>", unsafe_allow_html=True)

    # Define columns for layout
    col1, col2 = st.columns([3, 2], gap='medium')

    # Upload an image in the left column
    with st.sidebar:
        # Display the banner image
        st.image(banner_img)
        
        st.sidebar.header("Options")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"])
        st.divider()
        
    if uploaded_image is not None:
    
        with st.sidebar:
            # Sliders for confidence and overlap in the left column
            conf = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
            overlap = st.slider("Overlap Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            st.divider()
      
            if st.button("Detect Objects"):
            
                # Read the uploaded image with OpenCV
                image_bytes = uploaded_image.read()
                img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

                # Display a message
                message_placeholder = st.empty()
                message_placeholder.text("In progress... Please wait.")
                
                # Detect objects in the image
                pred_json = detect_objects(img, conf, overlap)
                if len(pred_json['predictions']) > 0:
                    if 'pred_json' in locals():
                        df_amount = calculate_amount(pred_json)
                        total_value = df_amount.Total.sum()
                    
                    # Get predictions and annotate the image
                    annotated_img = get_predictions(img, pred_json)
                    
                    message_placeholder.empty()   
                    
                    # Center column for displaying the annotated image
                    with col1:
                        st.subheader('Bills detected')
                        
                        st.text('Click to listen!')
                        # Read text
                        audio_bytes = get_audio(df_amount, total_value)
                        st.audio(audio_bytes) 
                            
                        # Display the annotated image
                        st.image(annotated_img, use_column_width=True, channels="BGR")

                    # Right column for displaying the summary table
                    with col2:
                        st.subheader('Amount Summary (in PHP)')
                      
                        # Display total value
                        st.markdown("<div style='background-color: #46008E; color: white; padding: 5px; text-align: center;'>" ##00FFCE, #CDFF00
                                    f"<p style='font-size: 24px; font-weight: bold; margin-top: 0; margin-bottom: 0;'>{total_value}</p>"
                                    "<p style='font-style: italic; font-size: small; margin-top: 0; margin-bottom: 0;'>Total Amount</p>"
                                    "</div>", unsafe_allow_html=True)
                                    
                        # Display the summary
                        st.dataframe(df_amount.drop(columns=['qty_words', 'value_num']), use_container_width=True, hide_index=True)
                        
                else:
                    st.error("No bills detected. Kindly adjust the confidence and overlap thresholds or provide a different image")
                       

if __name__ == "__main__":
    main()

import streamlit as st
import requests

st.title('Road infrastructure detection')

def main():
    image = st.file_uploader("detect infrastructure", type=['jpg', 'jpeg'])
    if st.button("detect") and image is not None:
        st.image(image)
        # send data and get the result
        response = requests.post("http://127.0.0.1:8000/clf_image", files={"file": image}).json()
        # print results
        for elem in range(len(response.get('class_indices', 0))):
          st.write(f"Detected item: {elem+1}, Class name: {response['class_names'][elem]}, class index: {response['class_indices'][elem]}, confidence: {response['confidences'][elem]}")
if __name__ == '__main__':
    main()
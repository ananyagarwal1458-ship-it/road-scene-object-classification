import streamlit as st

st.title("🚗 Road Scene Object Classification")

st.write("Welcome to my AI Road Scene Classification project!")

uploaded_file = st.file_uploader(
    "Upload a road-scene image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.success("Image uploaded successfully! ✅")

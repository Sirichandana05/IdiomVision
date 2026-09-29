import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import requests
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import base64

def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image_base64 = get_base64(r"D:\JN\background.jpeg")

st.markdown(f"""
    <style>
        .stApp {{
            background: url("data:image/png;base64,{bg_image_base64}") no-repeat center center fixed;
            background-size: cover;
            filter: brightness(80%);
        }}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

sbert = load_sbert_model()

st.title("📸 Idiom Similarity Finder")

st.subheader("Step 1: Upload Images")
uploaded_files = st.file_uploader("Upload up to 5 images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 5:
    st.success("✅ All 5 images uploaded! Proceed to the next step.")

    st.subheader("Step 2: Enter an Idiom")
    idiom = st.text_input("Enter an idiom:")

    if idiom and st.button("🔍 Predict"):
        st.subheader("Processing... Please wait!")

        def get_idiom_meaning(idiom_text):
            API_URL = "http://10.53.150.188:1234/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "llama-2-7b-chat",
                "messages": [
                    {"role": "system", "content": "You are an AI assistant that explains idioms in a single concise sentence."},
                    {"role": "user", "content": f"What does the idiom '{idiom_text}' mean?"}
                ],
                "temperature": 0.7
            }
            response = requests.post(API_URL, json=payload, headers=headers)
            return response.json()["choices"][0]["message"]["content"] if response.status_code == 200 else "Error fetching idiom meaning."

        def clean_meaning(response_text, idiom_text):
            response_text = re.sub(re.escape(idiom_text), "", response_text, flags=re.IGNORECASE)
            for phrase in ["The idiom", "means", "refers to", "is used to describe", "It means"]:
                response_text = response_text.replace(phrase, "").strip()
            return response_text.strip(" .,:;-")

        idiom_meaning = get_idiom_meaning(idiom)
        cleaned_meaning = clean_meaning(idiom_meaning, idiom)

        st.subheader("📖 Idiom Meaning:")
        st.markdown(f"<p style='color: white; font-size: 18px;'><b>{cleaned_meaning}</b></p>", unsafe_allow_html=True)

        #st.write(f"**{cleaned_meaning}**")

        idiom_embedding = sbert.encode([cleaned_meaning])

        #st.subheader("📝 Generated Captions and Similarity Scores:")
        similarities = []

        idiom_text_embedding = sbert.encode([idiom])


    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        caption_embedding = sbert.encode([caption])

        sim_to_meaning = cosine_similarity(caption_embedding, idiom_embedding)[0][0]
        sim_to_idiom_text = cosine_similarity(caption_embedding, idiom_text_embedding)[0][0]

    # Final score favors figurative (meaning) and penalizes literal (idiom wording)
        adjusted_score = sim_to_meaning - sim_to_idiom_text

        similarities.append((uploaded_file.name, image, caption, adjusted_score))


    similarities.sort(key=lambda x: x[3], reverse=True)

    st.subheader("🔹 Images Sorted by Relevance:")
    for rank, (name, img, caption, similarity) in enumerate(similarities, start=1):
        st.image(img, width=300)
        st.markdown(
            f"<p style='color: white; font-size: 16px; font-weight: bold;'>#{rank} - {name} (Similarity: {similarity:.4f})</p>",
            unsafe_allow_html=True
        )

        #st.image(img, caption=f"#{rank} - {name} (Similarity: {similarity:.4f})",width=300)
        #st.write(f"**Caption:** {caption}")

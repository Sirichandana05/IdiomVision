
import streamlit as st
import torch
import requests
import base64
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Idiom Vision", page_icon="📸", layout="wide")

def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image_base64 = get_base64("3409297.jpg")

st.markdown(f"""
    <style>
        /* Background Image with Reduced Brightness */
        .stApp {{
            background: url("data:image/png;base64,{bg_image_base64}") no-repeat center center fixed;
            background-size: cover;
            filter: brightness(80%); /* Adjust brightness (lower = darker) */
        }}

        /* Overlay to further darken background (optional) */
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.1); /* Dark overlay */
            z-index: -1;
        }}

         /* Title Styling */
        h1.main-title {{
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            color: #EE1D12 !important; 
            text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
            padding: 20px 0;
        }}
        h3.sub-title {{
            text-align: center;
            font-size: 22px;
            color: #EE1D12 !important;
        }}
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">📸 Idiom Vision</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-title">Analyze image sequences and idioms</h3>', unsafe_allow_html=True)

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

sbert_model = load_sbert_model()


def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def compute_similarity(text1, text2):
    embedding1 = sbert_model.encode(text1)
    embedding2 = sbert_model.encode(text2)
    return cosine_similarity([embedding1], [embedding2])[0][0]

def get_idiom_meaning(idiom_text):
    API_URL = "http://localhost:1234/v1/chat/completions"  
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama-2-7b-chat",  
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant that explains idioms in a single concise sentence without any additional details."
            },
            {"role": "user", "content": f"What does the idiom '{idiom_text}' mean?"}
        ],
        "temperature": 0.7
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Error fetching idiom meaning."

def clean_meaning(response_text, idiom_text):
    """Remove unnecessary words and idiom name from the definition"""
    unwanted_phrases = ["The idiom", "means", "refers to", "is used to describe", "It means"]
    
    for phrase in unwanted_phrases:
        response_text = response_text.replace(phrase, "").strip()

    response_text = response_text.replace(idiom_text, "").strip()
    
    return response_text

def analyze_sequence_type(seq_caption1, seq_caption2, idiom):
    """Determine sequence type based on similarity with cleaned idiom meaning."""
    idiom_meaning = get_idiom_meaning(idiom)
    cleaned_meaning = clean_meaning(idiom_meaning, idiom)  
    
    # Compute similarity for both sequence images
    similarity1 = compute_similarity(seq_caption1, cleaned_meaning)  
    similarity2 = compute_similarity(seq_caption2, cleaned_meaning)  
    
    # Average the similarity scores
    avg_similarity = (similarity1 + similarity2) / 2  

    # Determine type
    if avg_similarity < 0.05:
        sequence_type = "Irrelevant"
    elif avg_similarity < 0.2:
        sequence_type = "Idiomatic"
    else:
        sequence_type = "Literal"
    
    return sequence_type, cleaned_meaning, avg_similarity 


st.header("📷 Upload 6 Images (2 Sequence + 4 Candidates)")

col1, col2 = st.columns(2)
seq_img1 = col1.file_uploader("Upload 1st Sequence Image", type=["jpg", "jpeg", "png"])
seq_img2 = col2.file_uploader("Upload 2nd Sequence Image", type=["jpg", "jpeg", "png"])

col3, col4, col5, col6 = st.columns(4)
cand_img1 = col3.file_uploader("Upload Candidate Image 1", type=["jpg", "jpeg", "png"])
cand_img2 = col4.file_uploader("Upload Candidate Image 2", type=["jpg", "jpeg", "png"])
cand_img3 = col5.file_uploader("Upload Candidate Image 3", type=["jpg", "jpeg", "png"])
cand_img4 = col6.file_uploader("Upload Candidate Image 4", type=["jpg", "jpeg", "png"])

captions = {}

if all([seq_img1, seq_img2, cand_img1, cand_img2, cand_img3, cand_img4]):
    images = {
        "Sequence Image 1": Image.open(seq_img1).convert("RGB"),
        "Sequence Image 2": Image.open(seq_img2).convert("RGB"),
        "Candidate Image 1": Image.open(cand_img1).convert("RGB"),
        "Candidate Image 2": Image.open(cand_img2).convert("RGB"),
        "Candidate Image 3": Image.open(cand_img3).convert("RGB"),
        "Candidate Image 4": Image.open(cand_img4).convert("RGB"),
    }

    for img_name, img in images.items():
        st.image(img, caption=img_name,width=300)

    with st.spinner("Predicting the 3rd image"):
        for img_name, img in images.items():
            captions[img_name] = generate_caption(img)

    #st.subheader("📝 Generated Captions:")
    #for img_name, caption in captions.items():
    #    st.write(f"**{img_name}:** {caption}")

    #st.subheader("🔍 Similarity Scores:")
    seq_caption1 = captions["Sequence Image 1"]
    seq_caption2 = captions["Sequence Image 2"]
    candidate_captions = [
        ("Candidate Image 1", captions["Candidate Image 1"]),
        ("Candidate Image 2", captions["Candidate Image 2"]),
        ("Candidate Image 3", captions["Candidate Image 3"]),
        ("Candidate Image 4", captions["Candidate Image 4"]),
    ]

    with st.spinner("Computing cosine similarity..."):
        seq1_embedding = sbert_model.encode(seq_caption1)
        seq2_embedding = sbert_model.encode(seq_caption2)
        similarity_scores = []

        for cand_name, cand_caption in candidate_captions:
            cand_embedding = sbert_model.encode(cand_caption)
            
            # Compute similarity for both sequence images
            score1 = cosine_similarity([seq1_embedding], [cand_embedding])[0][0]
            score2 = cosine_similarity([seq2_embedding], [cand_embedding])[0][0]
            
            avg_score = (score1 + score2) / 2
            
            similarity_scores.append((cand_name, avg_score))
            # st.write(f"**{cand_name} Similarity Score:** {avg_score:.4f}")

    best_match = max(similarity_scores, key=lambda x: x[1])
    best_image_name = best_match[0]
    best_image_score = best_match[1]

    st.subheader("🎯 Predicted 3rd Image in Sequence")
    st.image(images[best_image_name], caption=f"Best Match: {best_image_name} (Score: {best_image_score:.4f})",width=300)

st.header("💡 Idiom Analysis")
idiom_input = st.text_input("Enter an idiom:")

if idiom_input and seq_img1 and seq_img2:
    with st.spinner("Fetching idiom meaning and analyzing sequence type..."):
        seq_caption1 = captions["Sequence Image 1"]
        seq_caption2 = captions["Sequence Image 2"]
        sequence_type, cleaned_meaning, avg_similarity_score = analyze_sequence_type(seq_caption1, seq_caption2, idiom_input)

    st.subheader("Idiom Meaning:")
    st.write(f"**{cleaned_meaning}**") 

    st.subheader("Sequence Type:")
    st.write(f"**The sequence is classified as:** {sequence_type}")

if sequence_type == "Irrelevant":
    st.warning("⚠️ The image sequence seems unrelated to the idiom.")

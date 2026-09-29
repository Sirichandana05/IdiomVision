# Idiom-Vision

## Introduction
- The main objective of the project is to improve the computational understanding of the idiomatic expressions using multimodal approaches.
- Integrate visual and textual information to distinguish between idiomatic and literal expressions.

## Subtask-A  
  - Participants will be presented with a set of 5 images and idiom word .
  - The goal is to rank the images according to how well they represent the sense in which the NC(Nominal Compound) is used in the given context sentence.


## Subtask-B
  - Participants will be given a target expression and an image sequence from which the last of 3
 images has been removed, and the objective will be to select the best fill from a set of
 candidate images.
 -  The NC sense being depicted (idiomatic or literal) will not be given, and this label should also be output.

## Dataset and Resources
<details>
  <summary>Data Set Components</summary>
  
  - Context sentences with nominal compounds (NCs) in idiomatic andliteral uses.
  - Image sets linked to each sentence.
  - Annotations indicating correct idiomatic or literal interpretation.
  - Descriptive captions for images (for participants not processing images directly)
</details>

<details>
  <summary>Data Availability</summary>
  
  - Training Dataset:
     - Available in English and Brazilian Portuguese for Subtask A.
     - Available in English only for Subtask B.
 - Evaluation Dataset:
     -  Same structure as training data.
     -  Released before the testing phase.
 - External Resources:
      - Participants may use additional data sources.
      - Must document any external resources clearly.
</details>

## Baseline Models
<details>
  <summary> Subtask- A</summary>
  
  - LlamA-2
  - BLIP-2
  - BERT
</details>

<details>
  <summary> Subtask- B</summary>
  
  - LlamA-2
  - BLIP-2
  - BERT
</details>

## Running the Project

Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Pranav-Venkat-YK/Idiom-Vision.git
```

Install all dependencies:
  ```bash
    pip install -r requirements.txt
```

Run the Streamlit app:

```bash
  streamlit run app.py
```

Make sure your server(llama.cpp or llama 2 model vis LM studio) is running at

```bash
  http://localhost:1234/v1/chat/completions
```

  

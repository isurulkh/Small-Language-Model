import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# Load model and tokenizer
model_path = "model/SmallDisMedLM.pt"
model = torch.load(model_path, map_location='cpu')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# Define UI elements with clear labels and spacing
st.title("Medical Chatbot 🩺")

st.header("Ask a question about your health:")
user_input = st.text_input("", placeholder="Type your query here...")

if st.button("Get Symptoms for a disease"):
    with st.spinner("Generating response..."):
        input_ids = tokenizer.encode(user_input, return_tensors='pt')
        output = model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            top_k=8,
            top_p=0.95,
            temperature=0.5,
            repetition_penalty=1.2
        )
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    st.success("Chatbot Response:")
    st.write(decoded_output)

st.markdown("---")
st.info("Disclaimer: This chatbot is not a substitute for professional medical advice. Always consult with a healthcare provider for diagnosis and treatment.")

st.sidebar.header("About")
st.sidebar.write("This chatbot is powered by a GPT-2 language model trained on medical text data. It aims to provide general information and advice, but it should not be relied upon for diagnosis or treatment.")

st.sidebar.header("Feedback")
st.sidebar.text_area("Please share any feedback or suggestions:")
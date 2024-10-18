#chatbot.py

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize chat history
chat_history_ids = None

# Function to generate a response from the bot
def chat_with_bot(user_input):
    global chat_history_ids

    # Encode the new user input
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input to the chat history
    if chat_history_ids is not None:
        chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        chat_history_ids = new_user_input_ids

    # Generate a response from the model
    response_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Get the model's response and decode it
    bot_response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

    return bot_response

# Streamlit UI setup
st.title("Chatbot with Streamlit")
st.write("Chat with the bot! Type your message below:")

# User input text box
user_input = st.text_input("You:", "")

if user_input:
    response = chat_with_bot(user_input)
    st.write(f"**Bot:** {response}")

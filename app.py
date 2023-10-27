import openai
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import asyncio
import numpy as np
import pandas as pd
import streamlit as st
import rf_ppl
import requests
import json
import prepro
import record_eeg
import gpt

prediction_label = gpt.predicted_label()
def main():

    st.title("🧠 Project.Budd-e()")
    
    # Use a container to keep chat history
    container = st.container()
    user_input = st.text_input("You: ")
    containerStat = st.container()

    if prediction_label == ' [Positive]':
        containerStat.write(f":green[{prediction_label}]")
    elif prediction_label == ' [Neutral]':
        containerStat.write(f":orange[{prediction_label}]")
    elif prediction_label == ' [Negative]':
        containerStat.write(f":red[{prediction_label}]")

    if st.button("Send"):
        # Display user message
        #container.write(f"You: {user_input}")
        
        # Get chatbot response
        response = gpt.chatbot_response(user_input)
        
        # Display chatbot message
        #container.write(f"{response}") 
        gpt.display_messages(container)

# Run the asynchronous event loop
if __name__ == "__main__":
    (main())
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import asyncio
import numpy as np
import pandas as pd
import streamlit as st
import chat

prediction_label = chat.predicted_label()
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
        response = chat.chatbot_response(user_input)
        
        # Display chatbot message
        #container.write(f"{response}") 
        chat.display_messages(container)

if __name__ == "__main__":
    (main())
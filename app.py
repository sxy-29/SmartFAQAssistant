import pandas as pd
import numpy as np
import openai
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = st.secrets["openai_key"]

# Load Data & Embeddings
try:
    df = pd.read_csv('qa_dataset_with_embeddings.csv')
    # Convert the string representation of the embeddings to numpy arrays
    df['Question_Embedding'] = df['Question_Embedding'].apply(
        lambda x: np.array(str(x).strip("[]").split(), dtype=float) if pd.notnull(x) else np.array([])
    )
    question_embeddings = np.stack(df['Question_Embedding'].values)
except FileNotFoundError:
    st.error("qa_dataset_with_embeddings.csv not found. Please upload the file.")
    st.stop()

# Embedding Model
def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def find_best_answer(user_question):
   # Generate embedding for the user's question
   user_question_embedding = get_embedding(user_question)

   # Calculate cosine similarity
   similarities = cosine_similarity([user_question_embedding], question_embeddings)[0]

   # Find the index of the most similar question
   most_similar_index = np.argmax(similarities)
   highest_similarity_score = similarities[most_similar_index]

   # Set a similarity threshold 
   similarity_threshold = 0.85 

   if highest_similarity_score >= similarity_threshold:
      most_relevant_answer = df.loc[most_similar_index, 'Answer']
      return most_relevant_answer
   else:
      return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?"

# Streamlit Interface
st.title("Heart, Lung, and Blood Health FAQ Assistant")
st.write("Ask a question about heart, lung, or blood-related health topics.")

user_question = st.text_input("Your question:")

# Implement the Question Answering Logic
if st.button("Get Answer"):
    if user_question:
      best_answer = find_best_answer(user_question)
      st.subheader("Answer:")
      st.write(best_answer)
    else:
        st.warning("Please enter a question.")

# Add a Clear button (Optional)
if st.button("Clear Question"):
    user_question = ""
    st.experimental_rerun() # Rerun the app to clear the input field

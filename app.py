from dotenv import load_dotenv
import streamlit as st
from backend.user_utils import *

if 'flag' not in st.session_state:
    st.session_state["flag"] = False
if 'tickets' not in st.session_state:
    st.session_state['tickets'] =" "
if 'department_value' not in st.session_state:
    st.session_state['department_value'] =" "

#Creating session variables
if 'HR_tickets' not in st.session_state:
    st.session_state['HR_tickets'] =[]
if 'IT_tickets' not in st.session_state:
    st.session_state['IT_tickets'] =[]
if 'Transport_tickets' not in st.session_state:
    st.session_state['Transport_tickets'] =[]



def main():
    load_dotenv()

    st.header("Automatic Ticket Classification Tool")
    #Capture user input
    st.write("We are here to help you, please ask your question:")
    user_input = st.text_input("🔍")

    if user_input:

        #creating embeddings instance...
        embeddings=create_embeddings()

        #Function to pull index data from Pinecone
        import os
        #We are fetching the previously stored Pinecome environment variable key in "Load_Data_Store.py" file
        index=pull_from_faiss(embeddings)
        
        #This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
        relavant_docs=get_similar_docs(index,user_input)

        #This will return the fine tuned response by LLM
        response=get_answer(relavant_docs,user_input)
        st.write(response)

        
        #Button to create a ticket with respective department
        button = st.button("Submit ticket?")

        if button:
            #Get Response
            st.session_state["flag"] = True

            embeddings = create_embeddings()
            query_result = embeddings.embed_query(user_input)

            #loading the ML model, so that we can use it to predit the class to which this compliant belongs to...
            department_value = predict(query_result)
            st.session_state['department_value'] = department_value
            st.write("your ticket has been sumbitted to : "+department_value)
            st.session_state['tickets'] = user_input
            #Appending the tickets to below list, so that we can view/use them later on...
            if department_value=="HR":
                st.session_state['HR_tickets'].append(user_input)
            elif department_value=="IT":
                st.session_state['IT_tickets'].append(user_input)
            else:
                st.session_state['Transport_tickets'].append(user_input)



if __name__ == '__main__':
    main()




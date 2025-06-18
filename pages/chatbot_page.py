import streamlit as st
from genai_services import answer_with_context
from chroma_services import query_documents

st.title("RAG QnA Chatbot")
st.write("Ask questions about your ingested document!")
user_query = st.chat_input("Your question:")
if user_query:
    # Query Chroma for context
    context_chunks = query_documents(user_query, n_results=3)

    with st.spinner("Generating answer..."):
        answer = answer_with_context(user_query, context_chunks)
    st.markdown(f"**Answer:** {answer}")
    st.expander("Show retrieved context").write("\n".join(context_chunks))

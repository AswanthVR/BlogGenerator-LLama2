import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers


# Function to get response from LLama 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    # LLama2 model initialization
    llm = CTransformers(model='model\llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.7})
    
    # Prompt Template
    template = """
       Write a blog in {blog_style} on the topic "{input_text}" with approximately {no_words} words. The blog should cover the following aspects:

    1. Introduction to the topic.
    2. Key points and details about the topic.
    3. Relevant examples or case studies.
    4. Challenges or considerations related to the topic.
    5. Conclusion summarizing the main points and providing insights or future directions.

    Ensure the blog is engaging, informative, and easy to understand.
    """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"],
                            template=template)
    
    # Generate the response from the LLama 2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

# Streamlit app configuration
st.set_page_config(page_title="Generate Blogs",
                    page_icon='📝',
                    layout='centered',
                    initial_sidebar_state='collapsed')

# Main content of the app
st.header("Generate Blogs using LLama 2 Model 🦙📝")

# Input fields
input_text = st.text_area("Enter the Blog Topic",height=80)

# Layout for additional fields
col1, col2 = st.columns([1, 2])
with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Select Blog Style',
                            ('Formal', 'Informal', 'Technical'), index=0)
    
# Generate button
submit = st.button("Generate")

# Handling the submission
if submit:
    if input_text.strip() == '' or no_words.strip() == '':
        st.error("Please fill out all fields.")
    else:
        #Show spinner while generating the response
        with st.spinner("Generating the Blog..."):
        # Display the generated response
            response = getLLamaresponse(input_text, no_words, blog_style)
            st.subheader("Generated Blog:")
            st.write(response)



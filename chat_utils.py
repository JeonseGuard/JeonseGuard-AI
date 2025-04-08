from langchain.prompts import ChatPromptTemplate

def print_relevant_documents(relevant_documents):
    for enum in enumerate(relevant_documents):
        index, relevant_documents = enum
        print(f'🔹 {index+1}번 연관 문서\n')    
        print(relevant_documents.page_content + '\n')

def join_relevant_documents(relevant_documents):
    return '\n\n'.join([d.page_content for d in relevant_documents])

def get_prompt():
    template = '''
    You are given legal text extracted via OCR along with related legal context.

    Your task is to determine whether the OCR text accurately reflects the legal meaning in the context.  
    Only analyze and explain **if there are legal inaccuracies or misinterpretations**.  
    If the OCR text is correct, you can briefly state that it is legally accurate. 
    
    Specifically, if issues are found:
    - Point out the inaccuracies or misinterpretations
    - Explain what the correct legal meaning should be

    Do not repeat the OCR Text or legal context in your answer.
    Only provide your analysis and conclusion.
    Respond in Korean.
    
    OCR Text:
    {text}

    Legal Context:
    {context}
    '''
    prompt = ChatPromptTemplate.from_template(template)
    return prompt
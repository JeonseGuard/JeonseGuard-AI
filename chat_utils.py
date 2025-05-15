from embedding_utils import get_embedding_model
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

def print_relevant_documents(relevant_documents):
    for enum in enumerate(relevant_documents):
        index, relevant_documents = enum
        print(f'🔹 {index+1}번 연관 문서\n')    
        print(relevant_documents.page_content + '\n')

def join_relevant_documents(relevant_documents):
    return '\n\n'.join([d.page_content for d in relevant_documents])

def get_prompt(template_name):
    clauses_template = '''
    당신에게 OCR로 추출된 법적 텍스트인 OCR Text와 추가로, 관련 법적 문맥인 Legal Context가 주어집니다.
    당신이 할 일은 OCR Text가 Legal Context의 법적 의미를 정확하게 반영하는지 확인하고 문제가 발견되면
    부정확하거나 잘못된 해석을 지적하고 올바른 해석에 대해 설명하는 것입니다.

    ### 이제 OCR Text와 Legal Context를 제공하겠습니다.
    OCR Text:
    {text}

    Legal Context:
    {context}

    ### 마지막으로 답변 형식을 제공하겠습니다. 우선 문제가 없는 경우 "조항에서 이상여부를 발견하지 못했습니다."라고만 출력하고
    이상여부가 발견된 경우에 대해서만 답변 해주세요. 답변 형식은 아래를 따라 주시고 각 형식 당 200자 내외로 친절하게 답변해주세요.
    * 제 O조: 이상이 있는 부분의 내용과 해석
    '''

    sc_template = '''
    아래의 특약사항을 요약하고 부연설명을 해주세요.

    ### 특약사항을 제공하겠습니다.
    {text}

    ### 마지막으로 답변 형식은 200자 내외로 친절하게 답변 해주세요.
    '''

    if template_name == 'clauses':
        prompt = ChatPromptTemplate.from_template(clauses_template)
    elif template_name == 'sc':
        prompt = ChatPromptTemplate.from_template(sc_template)
    return prompt

def get_clauses_response(ocr_text):
    embedding_model = get_embedding_model('openai')
    vectorstore_path = "./data/vectorstore"
    chat_model = "gpt-4o-mini"

    vectorstore = FAISS.load_local(folder_path = vectorstore_path,
                                   embeddings = embedding_model,
                                   allow_dangerous_deserialization=True)
    retreiver = vectorstore.as_retriever(search_type = 'mmr', search_kwargs = {'fetch_k': 150,
                                                                               'k': 30,
                                                                               'lambda_mult': 1})
    relevant_documents = retreiver.invoke(ocr_text)
    relevant_documents = join_relevant_documents(relevant_documents)

    prompt = get_prompt('clauses')
    llm = ChatOpenAI(model = chat_model,
                     temperature = 0,
                     max_tokens = 1000)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({'text': ocr_text, 'context': relevant_documents})
    
    return response

def get_sc_response(ocr_text):
    chat_model = "gpt-4o-mini"
    
    prompt = get_prompt('sc')
    llm = ChatOpenAI(model = chat_model,
                     temperature = 0,
                     max_tokens = 1000)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({'text': ocr_text})

    return response
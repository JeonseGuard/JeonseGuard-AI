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
    당신에게 OCR로 추출된 법적 조항 텍스트인 OCR Text와 추가로, OCR Text와 관련된 법적 문맥인 Legal Context가 주어집니다.
    Legal Context는 민법 조항, 주택임대차보호법 조항, 주택임대차보호법 시행령 조항을 포함하고 있습니다.
    OCR Text의 각 조와 Legal Text의 각 조는 같은 곳에서 발췌된 조항이 아니므로 서로 같은 내용이 아닐 수 있습니다.
    당신이 할 일은 Legal Context의 내용을 참조하여 OCR Text의 내용에서 부정확한 내용이 있는지 확인하고 발견하고 부정확한 내용이 발견되면 올바른 해석으로 설명하는 것입니다.

    ### 이제 OCR Text를 제공하겠습니다.
    OCR Text:
    {text}

    ### 다음으로 Legal Context를 제공하겠습니다.
    Legal Context:
    @민법
    {context1}

    @주택임대차보호법
    {context2}

    @주택임대차보호법 시행령
    {context3}

    ### 마지막으로 답변 형식을 제공하겠습니다.
    우선 부정확한 내용이 없는 경우 "조항에서 이상여부를 발견하지 못했습니다."라고만 답변 해주세요.
    부정확한 내용이 발견된 경우 답변 형식은 아래를 따라 주시고 각 형식 당 200자 내외로 친절하게 답변해주세요.
    * 제 O조: 부정확한 내용과 올바른 해석
    '''

    sc_template = '''
    당신에게 OCR로 추출된 텍스트인 특약사항이 주어집니다.
    당신이 할 일은 특약사항을 요악하고 도움이 될 만한 부연 설명을 덧붙여 설명하는 것입니다.

    ### 이제 특약사항을 제공하겠습니다.
    특약사항:
    {text}

    ### 마지막으로 답변 형식을 제공하겠습니다.
    답변 형식은 400자 내외로 친절하게 답변 해주세요.
    '''

    if template_name == 'clauses':
        prompt = ChatPromptTemplate.from_template(clauses_template)
    elif template_name == 'sc':
        prompt = ChatPromptTemplate.from_template(sc_template)
    return prompt

def get_clauses_response(ocr_text):
    embedding_model = get_embedding_model('openai')
    chat_model = "gpt-4o-mini"

    vectorstore_paths = ['./data/vectorstore/민법', './data/vectorstore/주택임대차보호법', './data/vectorstore/주택임대차보호법 시행령']
    vectorstore1 = FAISS.load_local(folder_path = vectorstore_paths[0], embeddings = embedding_model, allow_dangerous_deserialization=True)
    vectorstore2 = FAISS.load_local(folder_path = vectorstore_paths[1], embeddings = embedding_model, allow_dangerous_deserialization=True)
    vectorstore3 = FAISS.load_local(folder_path = vectorstore_paths[2], embeddings = embedding_model, allow_dangerous_deserialization=True)

    retreiver1 = vectorstore1.as_retriever(search_type = 'mmr', search_kwargs = {'fetch_k': 50,
                                                                                 'k': 10,
                                                                                 'lambda_mult': 1})
    retreiver2 = vectorstore2.as_retriever(search_type = 'mmr', search_kwargs = {'fetch_k': 40,
                                                                                 'k': 8,
                                                                                 'lambda_mult': 0.7})
    retreiver3 = vectorstore3.as_retriever(search_type = 'mmr', search_kwargs = {'fetch_k': 40,
                                                                                 'k': 8,
                                                                                 'lambda_mult': 0.7})

    relevant_documents1 = join_relevant_documents(retreiver1.invoke(ocr_text))
    relevant_documents2 = join_relevant_documents(retreiver2.invoke(ocr_text))
    relevant_documents3 = join_relevant_documents(retreiver3.invoke(ocr_text))

    prompt = get_prompt('clauses')
    llm = ChatOpenAI(model = chat_model, temperature = 0, max_tokens = 1000)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({'text': ocr_text, 'context1': relevant_documents1, 'context2': relevant_documents2, 'context3': relevant_documents3})

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

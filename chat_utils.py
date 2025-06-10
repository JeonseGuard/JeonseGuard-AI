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
    당신에게 앞으로 "임대차 계약서"의 조항들과 추가로, "민법"과 "주택 임대차 보호법"에서 발췌한 조항들이 주어집니다. 당신의 임무를 제공 하겠습니다.
    1. "임대차 계약서"의 내용을 꼼꼼하게 확인하세요. 임대인과 임차인을 헷갈리면 안됩니다.
    2. "민법"과 "주택 임대차 보호법"에서 발췌한 조항들을 참조하여 "임대차 계약서"의 내용을 확인하고 부정확한 내용이 있는 경우 올바르게 설명하세요.
    3. "민법"과 "주택 임대차 보호법"을 참조 할 때, 직접적인 연관이 없는 조항이나 중복되는 조항은 제외하고 참조하세요.
    4. "민법"의 조항들 중 "제565조"와 "제615조"는 각각 매매와 사용대차에서 발췌한 조항이지만 모두 계약서에 적용 가능한 조항들입니다.
    5. "임대차 계약서"의 조항을 확인할 때 완벽하게 잘못된 표현에만 집중하고, 나머지 내용에 대해서는 유연하게 대응하세요.

    # "임대차 계약서":
    {text}

    # "민법":
    {context1}

    # "주택 임대차 보호법":
    {context2}

    마지막으로 답변 형식을 제공하겠습니다.
    1. 부정확한 내용이 없는 경우의 답변은 "조항에서 이상여부를 발견하지 못했습니다."라고만 답변 해주세요.
    2. 부정확한 내용이 발견된 경우의 답변은 "제 O조: ..."의 형식을 따르고 각 형식당 150토큰(1~2줄 정도)내로 답변 해주세요. 반드시 부정확한 내용이 발견된 조항에 대해서만 적용 해주세요.
    '''

    sc_template = '''
    당신에게 "특약사항" 7개가 주어집니다. 
    당신이 할 일은 먼저 "특약사항"을 요악하고 도움이 될 만한 부연 설명을 덧붙여 설명하는 것입니다.

    #특약사항:
    {text}

    마지막으로 답변 형식을 제공 하겠습니다.
    1. "특약사항 요약입니다. ..."의 형식을 따라주세요.
    2. 답변은 400토큰(4~5줄 정도)내로 친절하게 답변 해주세요.
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
    #vectorstore3 = FAISS.load_local(folder_path = vectorstore_paths[2], embeddings = embedding_model, allow_dangerous_deserialization=True)

    retreiver1 = vectorstore1.as_retriever(search_type = 'mmr', search_kwargs = {'k': 3,
                                                                                 'lambda_mult': 0.7})
    retreiver2 = vectorstore2.as_retriever(search_type = 'mmr', search_kwargs = {'k': 3,
                                                                                 'lambda_mult': 0.7})
    #retreiver3 = vectorstore3.as_retriever(search_type = 'mmr', search_kwargs = {'k': 3,
    #                                                                             'lambda_mult': 0.7})

    queries = ['임차권 양도, 전대, 원상회복', '차임 연체, 해지', '임대인의 의무', '해약금, 해제', '이행지체, 손해배상', '임대차 기간']
    relevant_documents1 = ""
    for query in queries[:-1]:
        relevant_documents1 += join_relevant_documents(retreiver1.invoke(query))
    relevant_documents2 = join_relevant_documents(retreiver2.invoke(queries[-1]))

    prompt = get_prompt('clauses')
    llm = ChatOpenAI(model = chat_model, temperature = 0, max_tokens = 500)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({'text': ocr_text, 'context1': relevant_documents1, 'context2': relevant_documents2})

    print(relevant_documents1)
    print(relevant_documents2)

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

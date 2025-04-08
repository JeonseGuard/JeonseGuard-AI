from embedding_utils import get_embedding_model
from chat_utils import join_relevant_documents, get_prompt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
import json

with open('config.json', 'r', encoding = 'utf-8') as f:
    config = json.load(f)

embedding_model = get_embedding_model(config['embedding_model'])
vectorstore_path = config['vectorstore_path']
chat_model = config['chat_model']
# 추후 OCR로 검출한 조항으로 대체
ocr_text = '''
제 6조 (계약의 해제) 임차인이 임대인에게 중도금(중도금이 없을 때는 잔금)을 지불하기 전까지, 임대인은 계약 금의 배액을 상환하고, 임차인은 계약금을 포기하고 이 계약을 해제할 수 있다.
'''

vectorstore = FAISS.load_local(folder_path = vectorstore_path,
                               embeddings = embedding_model,
                               allow_dangerous_deserialization=True)
retreiver = vectorstore.as_retriever(search_type = 'mmr',
                                     search_kwargs = {'k': 15, 'lambda_mult': 0.8})
relevant_documents = retreiver.invoke(ocr_text)
relevant_documents = join_relevant_documents(relevant_documents)

prompt = get_prompt()
llm = ChatOpenAI(model = chat_model,
                 temperature = 0,
                 max_tokens = 1000)
chain = prompt | llm | StrOutputParser()
response = chain.invoke({'text': ocr_text, 'context': relevant_documents})
print(response)
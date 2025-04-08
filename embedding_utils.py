from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

def print_embeddings(embeddings):
    for enum in enumerate(embeddings):
        index, embedding = enum
        print(f'🔹 {index+1}번 임베딩 벡터(앞 차원 10개)\n')    
        print(embedding[:10])
        print()

def get_embedding_model(embedding_model):
    if(embedding_model == 'openai'):
        model = OpenAIEmbeddings() 
        return model
    elif(embedding_model == 'huggingface'):
        model = HuggingFaceEmbeddings(model_name = 'jhgan/ko-sroberta-nli',
                                      model_kwargs = {'device': 'cpu'},
                                      encode_kwargs = {'normalize_embedding': True})
        return model

def get_openai_embeddings(splits):
    model = get_embedding_model('openai')
    embeddings = model.embed_documents(splits)
    return embeddings

def get_huggingface_embeddings(splits):
    model = get_embedding_model('huggingface')
    embeddings = model.embed_documents(splits)
    return embeddings

def save_embeddings(splits, embeddings, embedding_model, vectorstore_path):
    model =  get_embedding_model(embedding_model)
    embeddings = list(zip(splits, embeddings))
    vectorstore = FAISS.from_embeddings(text_embeddings = embeddings,
                                        embedding = model,
                                        distance_strategy = DistanceStrategy.COSINE)
    vectorstore.save_local(vectorstore_path)
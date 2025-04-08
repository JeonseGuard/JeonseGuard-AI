from load_utils import get_contents
from split_utils import get_splits
from embedding_utils import print_embeddings, get_huggingface_embeddings, get_openai_embeddings, save_embeddings
import json

with open('config.json', 'r', encoding = 'utf-8') as f:
    config = json.load(f)

data_path = config['data_path']
file_type = config['file_type']
embedding_model = config['embedding_model']
vectorstore_path = config['vectorstore_path']

contents = get_contents(data_path, file_type)
splits = get_splits(contents)

if(embedding_model == 'openai'):
    embeddings = get_openai_embeddings(splits)
    print('💡 OpenAI 임베딩 결과...\n')
    print_embeddings(embeddings)
    save_embeddings(splits, embeddings, embedding_model, vectorstore_path)
elif(embedding_model == 'huggingface'):
    embeddings = get_huggingface_embeddings(splits)
    print('💡 HuggingFace 임베딩 결과...\n')
    print_embeddings(embeddings)
    save_embeddings(splits, embeddings, embedding_model, vectorstore_path)
from document_utils import get_document, get_splits
from embedding_utils import print_embeddings, get_huggingface_embeddings, get_openai_embeddings, save_embeddings
import os

documents_path = "./data/documents"
model_name = "openai"
vectorstore_path = "./data/vectorstore"
document_paths = [os.path.join(documents_path, document_name)
                  for document_name in os.listdir(documents_path)
                  if os.path.splitext(document_name)[1].lower() == '.pdf']

total_splits = []
for document_path in document_paths:
    document = get_document(document_path)
    splits = get_splits(document)
    for split in splits:
        total_splits.append(split)

if(model_name == 'openai'):
    embeddings = get_openai_embeddings(total_splits)
    print('💡 OpenAI 임베딩 결과...\n')
    print_embeddings(embeddings)
    save_embeddings(total_splits, embeddings, model_name, vectorstore_path)
elif(model_name == 'huggingface'):
    embeddings = get_huggingface_embeddings(total_splits)
    print('💡 HuggingFace 임베딩 결과...\n')
    print_embeddings(embeddings)
    save_embeddings(total_splits, embeddings, model_name, vectorstore_path)

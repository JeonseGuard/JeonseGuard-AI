from langchain_community.document_loaders import PyPDFLoader
import re
import os

def print_splits(splits):
    for idx, split in enumerate(splits):
        print(f'🔹 {idx+1}번 분할\n')    
        print(split + '\n')

def get_document(document_path):
    document_name = os.path.splitext(os.path.basename(document_path))[0]
    loader = PyPDFLoader(document_path)
    pages = loader.load()
    
    document = ""
    for page in pages:
        document += page.page_content
    document = document.strip()
    document = re.sub(fr'법제처\s+\d+\s+국가법령정보센터\n{document_name}', '', document)
    
    return document

def get_splits(document):
    splits = []

    contents = document.split('\n \n')[1:]
    contents = [re.sub(r'\s{2,}', ' ', content.replace('\n', ''))
                for content in contents]
    for content in contents:
        splits.append(content)
    
    return splits
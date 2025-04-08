from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
import os

def print_contents(contents):    
    for enum in enumerate(contents):
        index, content = enum
        print(f'🔹 {index+1}번 페이지\n')    
        print(content + '\n')

def get_contents(data_path, file_type):
    if(file_type == '*.pdf'):
        loader = DirectoryLoader(data_path, glob = file_type, loader_cls = PyPDFLoader)
    elif(file_type == '*.txt'):
        loader = DirectoryLoader(data_path, glob = file_type, loader_cls = TextLoader)
    pages = loader.load()
    contents = [page.page_content for page in pages]
    return contents
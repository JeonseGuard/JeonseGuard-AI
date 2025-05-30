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
    document = re.sub(r'부칙 \<[^>]+\>.*', '', document, flags = re.DOTALL) # 부칙 제거

    splits = document.split('\n \n')[2:-1]# 줄바꿈으로 나눠진 조항 분리
    splits = [split.strip() for split in splits] # 앞뒤 공백 제거
    splits = [re.sub(r' {2,}', ' ', split) for split in splits] # 2개 이상의 공백은 하나로 변경
    splits = [re.sub('\n', '', split) for split in splits] # 줄바꿈 제거
    splits = [split for split in splits if not re.match(r'제\d+[편장절관]', split)] # 편장절 제거
    splits = [re.sub(r'\<[^>]+\>', '', split) for split in splits] # 개정, 신설 등 보조문 제거
    splits = [re.sub(r'\[[^]]+\]', '', split) for split in splits] # 개정, 신설 등 보조문 제거
    
    return splits
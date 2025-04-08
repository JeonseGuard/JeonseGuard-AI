from langchain.text_splitter import RecursiveCharacterTextSplitter

def print_splits(splits):
    for enum in enumerate(splits):
        index, split = enum
        print(f'🔹 {index+1}번 분할\n')    
        print(split + '\n')

def get_splits(contents):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 700,
                                              chunk_overlap = 100,
                                              length_function = len,
                                              separators = ['\n\n', '\n', '.', ' '])
    splits = []
    for content in contents:
        chunks = splitter.split_text(content)
        for chunk in chunks:
            splits.append(chunk)
    return splits
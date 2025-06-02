from chat_utils import get_clauses_response, get_sc_response
import cv2
import re
import pytesseract as tesseract

def get_resized_image(image):
    h, w = image.shape[:2]
    resized_image = cv2.resize(image, (int(w*2.0), int(h*2.0)), interpolation = cv2.INTER_LINEAR)
    return resized_image

def get_dilated_image(image):
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bitwise_not(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated_image = cv2.dilate(gray_image, kernel, iterations = 2)
    dilated_image = cv2.bitwise_not(dilated_image)
    return dilated_image

def get_eroded_image(image):
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bitwise_not(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded_image = cv2.erode(gray_image, kernel, iterations = 1)
    eroded_image = cv2.bitwise_not(eroded_image)
    return eroded_image

def text_to_num(text):
    nums = {'일':1, '이':2, '삼':3, '사':4, '오':5, '육':6, '칠':7, '팔':8, '구':9}
    small_units = {'십':10, '백':100, '천':1000}
    large_units = {'만':10000, '억':100000000}
    
    result = 0
    tempResult = 0
    num = 0
    for char in text:
        if char in nums:
            num += nums[char]
        elif char in small_units:
            if num == 0: num = 1
            tempResult += num * small_units[char]
            num = 0
        elif char in large_units:
            tempResult += num
            if tempResult == 0: tempResult = 1
            result += tempResult * large_units[char]
            tempResult = 0
            num = 0
    result += tempResult + num
    result = f'{int(result):,}'
    
    return result

def get_ocr_config(psm, oem, c_list):
    config = f'--psm {psm} --oem {oem}'
    if c_list:
        config += ' ' + ' '.join([f'-c {c}' for c in c_list])
    return config

def get_ocr_result(image, lang, config):
    result = tesseract.image_to_string(image, lang = lang, config = config).strip()
    return result

def get_part_1_info(images, dict):
    # <----- 부동산 표시 부분(소재지, 용도, 임대할 부분) ----->
    config = get_ocr_config(7, 1, ['preserve_interword_spaces=1',
                                   'user_defined_dpi=300',
                                   'load_system_dawg=0',
                                   'load_freq_dawg=0'])
    
    result1 = get_ocr_result(images[0], 'kor', config)
    result2 = get_ocr_result(images[4], 'kor', config)
    result2 = ' '.join(reversed(result2.split()))
    result3 = get_ocr_result(images[2], 'kor', config)
    
    print(f'소재지: {result1}')
    print(f'임대할 부분: {result2}')
    print(f'구조, 용도: {result3}')

    pattern1 = r'(?P<address>[가-힣]+(시|도)\s[가-힣]+(시|군|구)(\s[가-힣]+(군|구))?\s[가-힣]+(읍|면|동)(\s[가-힣]+[리])?)\s(?P<bun>\d+)(-(?P<ji>\d+))?'
    pattern2 = r'(?P<hoName>\d+호\s?)?(?P<floorName>\d+층\s?)?(?P<dongName>[가-힣0-9,]+동)?'

    match = re.match(pattern1, result1)
    if match:
        dict["address"] = match.group('address')
        dict["bun"] = match.group('bun')
        if match.group('ji'):
            dict["ji"] = match.group('ji')
    match = re.match(pattern2, result2)
    if match:
        if match.group('dongName'):
            dict["dongName"] = match.group('dongName')
        if match.group('floorName'):
            dict["floorName"] = match.group('floorName')
        if match.group('hoName'):
            dict["hoName"] = match.group('hoName')
    dict["structUse"] = result3

    # <----- 부동산 표시 부분(면적) ----->
    config = get_ocr_config(7, 1, ['tessedit_char_whitelist=0123456789,.',
                                   'classify_bln_numeric_mode=1',
                                   'numeric_punctuation=1'
                                   'user_defined_dpi=300'])
    
    result1 = get_ocr_result(images[1], 'eng', config)
    result2 = get_ocr_result(images[3], 'eng', config)
    result3 = get_ocr_result(images[5], 'eng', config)

    print(f'토지 면적: {result1}')
    print(f'건물 면적: {result2}')
    print(f'임대할 면적: {result3}')

    dict["landArea"] = result1
    dict["bldgArea"] = result2
    dict["exclArea"] = result3
    
    # <----- 계약 내용 부분 ----->
    config = get_ocr_config(7, 1, ['preserve_interword_spaces=1',
                                   'user_defined_dpi=300',
                                   'load_system_dawg=0',
                                   'load_freq_dawg=0'])
    
    result1 = get_ocr_result(images[6], 'kor', config)
    result2 = get_ocr_result(images[7], 'kor', config)
    result3 = get_ocr_result(images[8], 'kor', config)
    result4 = get_ocr_result(images[9], 'kor', config)

    print(f'보증금: {result1}')
    print(f'계약금: {result2}')
    print(f'중도금: {result3}')
    print(f'잔금: {result4}')

    pattern1 = r'(금\s(?P<price>[가-힣]+)정)'
    pattern2 = r'(?P<date>\d+년\s\d+월\s\d+일)'

    match = re.match(pattern1, result1)
    if match: dict["deposit"] = f"{text_to_num(match.group('price'))}"
    match = re.match(pattern1, result2)
    if match: dict["downPayment"] = f"{text_to_num(match.group('price'))}"
    match = re.match(pattern1, result3)
    if match: dict["interimPayment"] = f"{text_to_num(match.group('price'))}"
    match = re.match(pattern1, result4)
    if match: dict["balance"] = f"{text_to_num(match.group('price'))}"
    match = re.search(pattern2, result3)
    if match: dict["interimDate"] = match.group('date')
    match = re.search(pattern2, result4)
    if match: dict["balanceDate"] = match.group('date')

    return dict

def get_part_2_info(images, dict):
    config = get_ocr_config(7, 1, ['preserve_interword_spaces=1',
                                   'user_defined_dpi=300',
                                   'load_system_dawg=0',
                                   'load_freq_dawg=0'])
    
    result1 = get_ocr_result(get_dilated_image(get_eroded_image(images[3])), 'kor', config)
    result2 = get_ocr_result(images[0], 'kor', config)
    result3 = get_ocr_result(get_dilated_image(get_eroded_image(images[7])), 'kor', config)
    result4 = get_ocr_result(images[4], 'kor', config)

    print(f'임대인 이름: {result1}')
    print(f'임대인 주소: {result2}')
    print(f'임차인 이름: {result3}')
    print(f'임차인 주소: {result4}')
    
    dict["lessorName"] = result1
    dict["lessorAddress"] = result2
    dict["lesseeName"] = result3
    dict["lesseeAddress"] = result4

    config = get_ocr_config(7, 1, ['tessedit_char_whitelist=0123456789-',
                                   'classify_bln_numeric_mode=1',
                                   'numeric_punctuation=1'
                                   'user_defined_dpi=300'])

    result1 = get_ocr_result(images[1], 'eng', config)
    result2 = get_ocr_result(images[5], 'eng', config)
    result3 = get_ocr_result(images[2], 'eng', config)
    result4 = get_ocr_result(images[6], 'eng', config)

    print(f'임대인 주민번호: {result1}')
    print(f'임대인 전화번호: {result2}')
    print(f'임차인 주민번호: {result3}')
    print(f'임차인 전화번호: {result4}')

    dict["lessorId"] = result1
    dict["lesseeId"] = result2
    dict["lessorPhone"] = result3
    dict["lesseePhone"] = result4

    return dict

def get_part_3_info(images, dict):
    config = get_ocr_config(6, 1, ['preserve_interword_spaces=1',
                                   'user_defined_dpi=300',
                                   'load_system_dawg=0',
                                   'load_freq_dawg=0'])
    
    result1 = get_ocr_result(images[0], 'kor', config)
    result2 = get_ocr_result(images[1], 'kor', config)

    print(f'조항: {result1}')
    print(f'특약사항: {result2}')

    dict["clausesResponse"] = get_clauses_response(result1)
    dict["scResponse"] = get_sc_response(result2)
    
    return dict
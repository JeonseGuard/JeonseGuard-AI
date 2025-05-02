import os
from dotenv import load_dotenv
from openai import OpenAI
import certifi #SSL 인증서 문제 해결을 위한 라이브러리

# SSL 인증서 경로 설정
os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

API_KEY = os.environ["API_KEY"]
SYSTEM_MESSAGE = os.environ["SYSTEM_MESSAGE"]

BASE_URL = "https://api.together.xyz"
MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

messages = [
    {
        'role': 'system', 
        'content': SYSTEM_MESSAGE
    },
]

print("챗봇을 시작합니다. 종료하려면 quit 또는 exit를 입력하세요.")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit"]:
        print("챗봇을 종료합니다.")
        break

    messages.append({
        'role': 'user',
        'content': user_input
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7, #창의성
        # max_tokens=1000,
        # top_p=1.0,
        # frequency_penalty=0.0,
        # presence_penalty=0.0
    )

    chatbot_reply = response.choices[0].message.content
    print(f"Chatbot: {chatbot_reply}")

    messages.append({
        'role': 'assistant',
        'content': chatbot_reply
    })
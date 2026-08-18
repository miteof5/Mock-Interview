from openai import OpenAI
from interview_agent.config import get_settings

settings = get_settings()

client = OpenAI(
    api_key=settings.api_key,
    base_url=settings.dashscope_base_url
)

def test_llm():
    print("🚀正在调用通义千问接口……")
    resp = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content":"只用一句话回复：接口测试成功"}]
    )
    print("✅调用成功！返回结果：")
    print(resp.choices[0].message.content)

if __name__ == "__main__":
    test_llm()

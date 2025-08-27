from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate


api_key = "Enter API Key Here"

# 1) Initialize Groq LLM via LangChain
llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile"   # you can change to another Groq model
)

# 2) Create a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "How to Treat Lung Cancer.")
])

# 3) Run the chain
chain = prompt | llm
response = chain.invoke({})

print("\n✅ Groq LangChain Response:\n")
print(response.content)

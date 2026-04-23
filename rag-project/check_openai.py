from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
# LangSmith auto-activates from .env:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=...
#   LANGCHAIN_PROJECT=...

llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("Answer briefly: {question}")
chain  = prompt | llm | StrOutputParser()

response = chain.invoke({"question": "What top 5 professions will earn more in next 5-10 years?"})
print(response)

# cmd-o/p:
# The top 5 professions likely to earn more in the next 5-10 years include:

# 1. **Healthcare Professionals** (e.g., nurses, physicians, telehealth specialists)
# 2. **Technology Specialists** (e.g., AI/ML engineers, cybersecurity experts)
# 3. **Data Analysts/Scientists** (due to increasing data reliance)
# 4. **Renewable Energy Technicians** (as green energy demand grows)
# 5. **Financial Analysts/Advisors** (with evolving financial markets and investment strategies)
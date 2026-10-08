import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

api_key=str(os.getenv("API_KEY"))
base_url=str(os.getenv("BASE_URL"))

class PersonInfo(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")

parser = PydanticOutputParser(pydantic_object=PersonInfo)

model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    base_url=base_url,
    temperature=0.7,
)

prompt = f"""
Extract the person's info from this text:
"Riya is 29 years old."

{parser.get_format_instructions()}
"""

response = model.invoke(prompt)
result = parser.parse(response.content)

print()
print("Pydantic structured response")
print(result)
print(result.name)
print(result.age)

# JSON response
json_parser = JsonOutputParser()

new_prompt = f"""
Extract the person's info from this text:
"Riya is 29 years old."

Return the result as JSON with these keys:
name, age

{json_parser.get_format_instructions()}
"""

response = model.invoke(new_prompt)
result = json_parser.parse(response.content)

print()
print("JSON response")
print(result)
print(result["name"])
print(result["age"])

# Bad JSON response
correct_parser = JsonOutputParser()

correct_prompt = f"""
Extract the person's info from this text:
"Riya is 29 years old."

Return only valid JSON.
Use exactly these keys:
"name", "age"

{correct_parser.get_format_instructions()}
"""

response = model.invoke(correct_prompt)
print()
print("correct bad json")
print(response.content)
result = correct_parser.parse(response.content)
print(result)

# try except
try_parser = JsonOutputParser()

try_prompt = f"""
Extract the person's info from this text:
"Riya is 29 years old."

Return only valid JSON with these keys:
"name", "age"

{try_parser.get_format_instructions()}
"""
try:
    response = model.invoke(try_prompt)
    result = try_parser.parse(response.content)

    print()
    print("Parsed successfully:")
    print(result)
    print(result["name"])
    print(result["age"])

except Exception as e:
    print("Parsing failed.")
    print(f"Error: {e}")

#  retry logic
try_prompt = f"""
Extract the person's info from this text:
"Riya is 29 years old."

Return only valid JSON with these keys:
"name", "age"

{try_parser.get_format_instructions()}
"""

try:
    response = model.invoke(try_prompt)
    result = try_parser.parse(response.content)

except Exception:
    bad_output = response.content

    retry_prompt = f"""
The following output was not valid JSON:

{bad_output}

Please rewrite it as valid JSON only.
Use exactly these keys:
"name", "age"
"""

    retry_response = model.invoke(retry_prompt)
    result = try_parser.parse(retry_response.content)

print()
print("retry logic")
print(result)
import json
import openai
import pandas as pd
from tqdm.auto import trange
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from content import MOVIELIST
from content import OPENAI_API_BASE, OPENAI_API_TYPE, OPENAI_API_VERSION, OPENAI_API_KEY, OPENAI_DEPLOYMENT_NAME

openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE
openai.api_type = OPENAI_API_TYPE
openai.api_version = OPENAI_API_VERSION
chat = ChatOpenAI(engine=OPENAI_DEPLOYMENT_NAME, openai_api_key=OPENAI_API_KEY)

# 从prompts.txt文件中提取input要求
with open("data/prompts.txt", "r", encoding="utf-8") as file:
    content = file.read()
prompts = content.split("———————————————————————————————————————")
prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
input_string = prompts[0] # 0-剧情节奏, 1-改编来源, 2-主题, 3-情感基调
theme_list = input_string.split("参考列表：")[1].split("]")[0].replace("[", "").strip()
theme_output = input_string.split("输出json格式，格式参考")[1].split(".")[0].strip()
theme_template = '\n候选列表：{theme_list}\n输出：\n' + input_string.split("输出：")[1].replace(theme_output, '{output}').strip() + "\nanswer the users question as best as possible.\n{format_instructions}"

# 配置response_schemas模版
response_schemas = [
    ResponseSchema(name="1.generate one tag", description="should be a string"),
    ResponseSchema(name="2.repeat 1 for 3 times", description=""),
    ResponseSchema(name="3.judgement", description=""),
    ResponseSchema(name="4.filter tags score and rank", description=""),
    ResponseSchema(name="5.output", description="should be a json format"),
]

if __name__ =='__main__':
    
    df = pd.DataFrame()
    for movie in MOVIELIST:
        target_movie = movie
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        system_message_prompt = SystemMessagePromptTemplate.from_template(theme_template)
        human_template="目标电影：{target_movie}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        prompt = ChatPromptTemplate(
            messages=[system_message_prompt, human_message_prompt],
            input_variables=["theme_list", "target_movie", "output"],
            partial_variables={"format_instructions": format_instructions}
        )

        _input = prompt.format_prompt(theme_list=theme_list, target_movie=target_movie, output=theme_output)
        output = chat(_input.to_messages())
        output = output_parser.parse(output.content)
        
        try:
            data = json.loads(output['5.output'])
        except:
            data  = output['5.output'] 
        df = pd.concat([df, pd.DataFrame([data])])

    df.to_csv("output.csv") # 输出结果导出
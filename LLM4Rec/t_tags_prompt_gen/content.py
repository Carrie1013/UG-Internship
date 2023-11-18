import pandas as pd

OPENAI_API_BASE= "https://tcl-ai.openai.azure.com"
OPENAI_API_TYPE = "azure"
OPENAI_API_VERSION = '2023-07-01-preview'
OPENAI_API_KEY= "6e25ec6fa59d44f8af091db59e6db6d7"
OPENAI_DEPLOYMENT_NAME = 'gpt-4-0314'

MOVIELIST = pd.read_csv('data/movie_tags.csv').head(20)['title'].tolist() # 取movie_tags数据集合中的前20个媒资 - 只取title, prompt中要求基于data source判断judgment步骤

theme_output= "{“目标电影”:“”, “标签”:“”,  “得分”:“”}"
theme_list = "节奏缓慢，渐进节奏，节奏紧凑，剧情反转，紧张刺激，高潮迭起，悬念迭起，悬疑重重，回溯叙述，思考深度，情感共鸣，多线索，节奏明快, 轻松幽默"

theme_template= '''
    候选列表：{theme_list}
    输出：剧情节奏
    
    请逐步思考
    1.参考候选列表，结合你的知识(豆瓣,百度百科,维基百科等)，生成输出目标电影的剧情节奏，如果你不知道，请返回‘其他’
    2.重复上述操作三次。
    3.请针对上述所有标签，逐个判断是否有明确依据，输出符合程度‘高’，‘中’，‘低’。
    4.仅保留符合程度为‘高’的标签，根据依据来源投票打分并排序（10分制），保留大于9分的标签；如果没有大于9分的标签，则仅保留top1的标签。
    5.输出json格式，格式参考{output}。
    6.如果步骤5生成的结果中，参考列表未出现且没有相似标签时，请添加到参考列表，并返回最新的参考列表
    answer the users question as best as possible.
    {format_instructions}
'''
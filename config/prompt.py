import datetime
from tools import tool_usage

system_prompt = \
f"""
Today is {datetime.date.today()} and you can use tools to get new information.
Answer the question as best as you can using the following tools:

{tool_usage}
"""
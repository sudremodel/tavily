from flask import Flask, request, jsonify
import json
import time
import requests
from config import OPENAI_API_KEY, TAVILY_API_KEY
from openai import OpenAI
from tavily import TavilyClient

app = Flask(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
assistant_instruction = """You an analyst specializing in open-source intelligence, 
Your role is to gather and analyze publicly available information for market research and competitive analysis. 
You will provide insights, trends, and data-driven answers.
Never use your own knowledge to answer questions.
Always include the relevant urls for the sources you got the data from."""
assistant = client.beta.assistants.create(
    instructions=assistant_instruction,
    model="gpt-4-1106-preview",
    tools=[{
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Get information on recent events from the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to use. For example: 'Provide a competitive analysis of Open Source survey tools'"},
                },
                "required": ["query"]
            }
        }
    }]
)

def tavily_search(query):
    search_result = tavily_client.get_search_context(query, search_depth="advanced", max_tokens=8000)
    return search_result

def submit_tool_outputs(thread_id, run_id, tools_to_call):
    tool_output_array = []
    for tool in tools_to_call:
        output = None
        tool_call_id = tool.id
        function_name = tool.function.name
        function_args = tool.function.arguments

        if function_name == "tavily_search":
            output = tavily_search(query=json.loads(function_args)["query"])

        if output:
            tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

    return client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_output_array
    )

def wait_for_run_completion(thread_id, run_id):
    while True:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        print(f"Current run status: {run.status}")
        if run.status in ['completed', 'failed', 'requires_action']:
            return run

# Print messages from a thread
def print_messages_from_thread(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    message_list = []
    for msg in messages:
        message_list.append(f"{msg.role}: {msg.content[0].text.value}")
    return message_list

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_input = data['input']

    # Create a thread
    thread = client.beta.threads.create()

    # Create a message
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input,
    )

    # Create a run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    # Wait for run to complete
    run = wait_for_run_completion(thread.id, run.id)

    if run.status == 'failed':
        return jsonify({'error': run.error}), 500
    elif run.status == 'requires_action':
        run = submit_tool_outputs(thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls)
        run = wait_for_run_completion(thread.id, run.id)

    # Print messages from the thread
    messages = []
    messages = print_messages_from_thread(thread.id)

    return jsonify({'messages': messages})

if __name__ == '__main__':
    app.run(debug=True)

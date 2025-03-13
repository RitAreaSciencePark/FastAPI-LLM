from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel
from typing import List
import ollama
import json
import re
import os

debug = True
from elab_vector import ElabVector

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount the "static" folder so it’s accessible at the "/static" URL path
app.mount("/static", StaticFiles(directory="static"), name="static")

elab_vector = ElabVector()

# Conversation Memory Model
class WorkingMemory(BaseModel):
    role: str
    content: str

class MemoryFoam(BaseModel):
    messages: List[WorkingMemory] = []

# Initialize conversation memory
conversation_memory = MemoryFoam(messages=[])

def build_contextual_messages(user_query: str, memory: List[WorkingMemory], max_chars: int = 4000) -> List[dict]:
    """
    Dynamically build a message list for the LLM, including recent memory
    without exceeding a character limit.
    """
    # Base system prompt
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant that can answer general questions. "
                   "If asked about experiment or experiments, return ONLY [EXP]. "
                   "Answer normally otherwise."
    }

    # Start building the context in reverse (most recent first)
    context = []
    total_chars = 0

    # Loop from the end (most recent) backwards
    for message in reversed(memory):
        msg_dict = {"role": message.role, "content": message.content}
        message_len = len(message.content)

        if total_chars + message_len > max_chars:
            break  # Stop if adding this message would exceed the limit

        context.insert(0, msg_dict)  # Add to the beginning of the list
        total_chars += message_len

    # Finally, add the current user query
    context.append({"role": "user", "content": user_query})

    # Prepend system message
    return [system_message] + context



# -- Utility / logic functions (same as your original code) --
def is_experiment_request(query: str) -> bool:
    lowered = query.lower()
    return "experiment" in lowered or "experiments" in lowered

def remove_html_tags(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)

def clean_body(text: str) -> str:
    text = text.replace("\n", " ")
    text = remove_html_tags(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean(retrieved_texts):
    cleaned_entries = []
    for entry in retrieved_texts:
        if "Body:" in entry:
            prefix, body = entry.split("Body:", 1)
            cleaned_body = clean_body(body)
            cleaned_entry = f"{prefix}Body: {cleaned_body}"
            cleaned_entries.append(cleaned_entry)
        else:
            cleaned_entries.append(entry)
    return cleaned_entries

@app.get("/", response_class=HTMLResponse)
async def redirect_to_query():
    """
    Redirect dalla home alla pagina del form (/query/).
    """
    return RedirectResponse(url="/query/")

@app.get("/query", response_class=HTMLResponse)
async def get_query_form(request: Request):
    """
    Mostra il form per inserire la query.
    """
    return templates.TemplateResponse(
        "response.html",
        {
            "request": request,
            "user_query": "",
            "llm_input": "",
            "assistant_reply": "",
            "show_sse": False,
            "debug": debug,
        }
    )

@app.post("/query", response_class=HTMLResponse)
async def query_llm(request: Request, content: str = Form(...)):
    """
    Main endpoint. If experiments are needed and we don't have a good existing summary,
    we let the front-end do SSE streaming, then finalize automatically at the end.
    """
    print(f"[User Query]: {content}")
    conversation_memory.messages.append(WorkingMemory(role="user", content=content))

    # 1) Normal LLM check
    normal_answer = normal_llm_answer(content)
    conversation_memory.messages.append(WorkingMemory(role="assistant", content=normal_answer))

    # If not about experiments, just return...
    if "[EXP]" not in normal_answer:
        return templates.TemplateResponse(
            "response.html",
            {
                "request": request,
                "user_query": content,
                "llm_input": content,
                "assistant_reply": normal_answer,
                "show_sse": False,
                "debug": debug,
            }
        )

    # 2) The user asked about experiments
    summary_path = "summaries.json"
    existing_summary = ""
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                existing_summary = f.read()
        except Exception as e:
            print(f"Could not read summaries.json: {e}")

    if existing_summary.strip():
        uncertain = ask_llm_with_summary(content, existing_summary)
        if is_model_unsure(uncertain):
            query_summary_path = "query_summary.json"
            existing_query_summary = ""
            if os.path.exists(query_summary_path):
                try:
                    with open(query_summary_path, "r") as f:
                        existing_query_summary = f.read()
                    uncertain = ask_llm_with_summary(content, existing_query_summary)
                except Exception as e:
                    print(f"Could not read query_summary.json: {e}")
        if not is_model_unsure(uncertain):
            conversation_memory.messages.append(WorkingMemory(role="assistant", content=uncertain))
            return templates.TemplateResponse(
                "response.html",
                {
                    "request": request,
                    "user_query": content,
                    "llm_input": existing_summary,
                    "assistant_reply": uncertain,
                    "show_sse": False,
                    "debug": debug,
                }
            )

        print("[LLM was unsure, proceeding to SSE streaming...]")

    # 3) No existing summary or it's insufficient -> show SSE
    placeholder_reply = (
        "I'm gathering the lab experiments in real time now. Please wait while "
        "the streaming summaries are generated below..."
    )
    conversation_memory.messages.append(WorkingMemory(role="assistant", content=placeholder_reply))

    return templates.TemplateResponse(
        "response.html",
        {
            "request": request,
            "user_query": content,
            "llm_input": "[No suitable summary found. We'll stream a new one...]",
            "assistant_reply": placeholder_reply,
            "show_sse": True,
            "debug": debug,

        }
    )

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def normal_llm_answer(user_query: str) -> str:
    """
    LLM response using recent memory for non-experiment questions.
    """
    client = ollama.Client(host='http://10.128.2.165:11434')
    messages = build_contextual_messages(user_query, conversation_memory.messages)

    response = client.chat(
        model='llama3.3:latest',
        messages=messages
    )
    return response["message"]["content"]
def ask_llm_with_summary(user_query: str, summary_text: str) -> str:
    """
    Use conversation memory and experiment summary to answer the user query.
    """
    client = ollama.Client(host='http://10.128.2.165:11434')

    # Rebuild partial memory context
    context = build_contextual_messages(user_query, conversation_memory.messages)

    # We’ll combine the experiment summary and the user query into one message
    summary_prompt = (
        f"You are an AI assistant with lab data summaries.\n\n"
        f"SUMMARY OF EXPERIMENTS:\n{summary_text}\n\n"
        f"USER QUERY:\n{user_query}\n\n"
        "If this summary clearly answers the user's question, provide the answer. "
        "If you are NOT sure, respond with 'NOT SURE'. If the user ask to search better, respond with 'NOT SURE'"
    )

    # Replace the final user message in the context with the new structured one
    context[-1]["content"] = summary_prompt

    # Replace system message
    context[0] = {
        "role": "system",
        "content": "You are an AI assistant with lab data."
    }

    response = client.chat(
        model='llama3.3:latest',
        messages=context
    )
    return response["message"]["content"]


def is_model_unsure(llm_response: str) -> bool:
    return "NOT SURE" in llm_response.upper()

@app.get("/sse_page", response_class=HTMLResponse)
def sse_page(request: Request):
    return templates.TemplateResponse("stream_map.html", {"request": request})

@app.get("/stream_map")
def stream_map(query: str = ""):
    """
    Endpoint that 'streams' partial summaries while building them.
    Client uses SSE to update the page in real time.
    """
    # We define a generator function to yield data line-by-line:
    async def event_stream(query: str):
        client = ollama.Client(host='http://10.128.2.165:11434')

        data = elab_vector.raw_data  # or wherever your experiments are stored
        individual_summaries = []

        for exp in data:
            # Build your prompt
            if query:
                prompt = f"""
                Summarize the following experiment answering this query "{query}" in 2 lines max.
                ID: {exp['id']}
                Title: {exp['title']}
                Author: {exp['fullname']}
                Body: {exp['body']}
                """
            else:
                prompt = f"""
                Summarize the following experiment in 2 lines max.
                ID: {exp['id']}
                Title: {exp['title']}
                Body: {exp['body']}
                """

            # Call Ollama
            response = client.generate(model='llama3.3:latest', prompt=prompt, stream=False)
            summary_text = response['response'].strip()

            # Store the result
            individual_summaries.append({
                'id': exp['id'],
                'summary': summary_text
            })

            # Yield partial result to the client as SSE
            partial_json = json.dumps({"id": exp["id"], "summary": summary_text})
            yield f"data: {partial_json}\n\n"

            # Optional: small sleep to illustrate streaming
            #await asyncio.sleep(0.2)

        # After building them all, write the JSON file
        summaries_json = json.dumps(individual_summaries, indent=2)
        filename = 'query_summary.json' if query else 'summaries.json'
        with open(filename, "w") as json_file:
            json_file.write(summaries_json)

        # Finally, yield a "done" message so the client knows to stop
        yield "data: {\"done\": true}\n\n"

    # Return a StreamingResponse, telling it to use SSE
    return StreamingResponse(
        event_stream(query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.get("/finalize_answer")
def finalize_answer(query: str):
    """
    Once the SSE streaming has completed and query_summary.json is created,
    we read it, pass it to the LLM, and get a final answer.
    Returns JSON with { "answer": "..."}.
    """
    # 1) Read the newly created file
    query_summary_path = "query_summary.json"
    with open(query_summary_path, "r") as f:
        new_summary = f.read()

    # 2) Ask the model for the final answer (with the fresh summary)
    final_answer = ask_llm_with_summary(query, new_summary)
    conversation_memory.messages.append(WorkingMemory(role="assistant", content=final_answer))

    # 3) Return as JSON
    return {"answer": final_answer}

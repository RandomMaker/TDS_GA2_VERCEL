from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from students_data import student
from typing import Dict, List
import numpy as np
import requests
import re
import json


app = FastAPI()

AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDI2MzVAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.gUohb1yh4JYgWzv_JqpPXpXiRRELwbRgnvolbpVX3DI"

marks = [
    {"name": "Vr", "marks": 12},
    {"name": "oCq", "marks": 61},
    {"name": "AS74", "marks": 93},
    {"name": "zj8BtF", "marks": 43},
    {"name": "CMr", "marks": 5},
    {"name": "UG6", "marks": 52},
    {"name": "z7h", "marks": 6},
    {"name": "XOFeP8c", "marks": 25},
    {"name": "mu", "marks": 37},
    {"name": "vY2K", "marks": 96},
    {"name": "sb", "marks": 10},
    {"name": "fKnFVS6qP", "marks": 6},
    {"name": "D72MSJVZt", "marks": 44},
    {"name": "3BSGdTQ", "marks": 45},
    {"name": "Hg9vd", "marks": 12},
    {"name": "cR", "marks": 66},
    {"name": "iU4GZdIa", "marks": 2},
    {"name": "dpf", "marks": 65},
    {"name": "4F", "marks": 64},
    {"name": "XG9k226EN", "marks": 55},
    {"name": "Sq78U08LRz", "marks": 46},
    {"name": "61fz", "marks": 48},
    {"name": "Q3Lw4Z7", "marks": 14},
    {"name": "kntW", "marks": 97},
    {"name": "hM", "marks": 27},
    {"name": "Wxbt", "marks": 38},
    {"name": "7", "marks": 13},
    {"name": "UY6Y9", "marks": 85},
    {"name": "YBT", "marks": 75},
    {"name": "1u", "marks": 70},
    {"name": "Wz4dkt", "marks": 29},
    {"name": "tgS9Tu4L", "marks": 36},
    {"name": "lH1", "marks": 16},
    {"name": "FMT", "marks": 18},
    {"name": "s729w", "marks": 10},
    {"name": "KS", "marks": 37},
    {"name": "Swmd", "marks": 89},
    {"name": "DH7pW", "marks": 87},
    {"name": "LY6", "marks": 20},
    {"name": "ylSjM1y7w", "marks": 31},
    {"name": "h", "marks": 11},
    {"name": "wgHx", "marks": 25},
    {"name": "2sDUqdW", "marks": 0},
    {"name": "TL", "marks": 89},
    {"name": "2", "marks": 13},
    {"name": "Xw9qNC", "marks": 26},
    {"name": "cYS74B", "marks": 0},
    {"name": "5eA", "marks": 38},
    {"name": "m1JGT", "marks": 25},
    {"name": "Hw1eqQ", "marks": 38},
    {"name": "G", "marks": 51},
    {"name": "8", "marks": 57},
    {"name": "Pj5", "marks": 18},
    {"name": "i8Xq", "marks": 94},
    {"name": "ZnhMHPDb0U", "marks": 14},
    {"name": "P", "marks": 40},
    {"name": "DNVDIR", "marks": 61},
    {"name": "kkjcpNG", "marks": 73},
    {"name": "idazJQjb", "marks": 69},
    {"name": "RWw", "marks": 91},
    {"name": "LiJqr", "marks": 17},
    {"name": "2F4Zg2LMAQ", "marks": 50},
    {"name": "kOnaD", "marks": 41},
    {"name": "hBOxjmP", "marks": 78},
    {"name": "6", "marks": 14},
    {"name": "9Dcl", "marks": 39},
    {"name": "UNfCl9vNQ", "marks": 6},
    {"name": "X", "marks": 42},
    {"name": "QKbg", "marks": 17},
    {"name": "lL2", "marks": 9},
    {"name": "wRYmxTg20", "marks": 57},
    {"name": "skJ5", "marks": 28},
    {"name": "kiEOxsB", "marks": 43},
    {"name": "ctet", "marks": 60},
    {"name": "w9zXgjL", "marks": 93},
    {"name": "l6", "marks": 90},
    {"name": "us8q8y0ns", "marks": 97},
    {"name": "EAdR", "marks": 22},
    {"name": "4hzREAn", "marks": 25},
    {"name": "5WEy6g2", "marks": 96},
    {"name": "WVuwuWz", "marks": 57},
    {"name": "NRrHThD", "marks": 80},
    {"name": "kgqeSnq", "marks": 16},
    {"name": "m", "marks": 43},
    {"name": "VP", "marks": 44},
    {"name": "gUEfX7vodF", "marks": 8},
    {"name": "vZ4u9nF", "marks": 59},
    {"name": "vWsp2P59B", "marks": 11},
    {"name": "lZsq", "marks": 50},
    {"name": "cGq9a87ZP", "marks": 6},
    {"name": "zP", "marks": 46},
    {"name": "xslYOaYJ0", "marks": 47},
    {"name": "97M", "marks": 99},
    {"name": "vYUwN79tbr", "marks": 59},
    {"name": "GRt5QLzlK", "marks": 61},
    {"name": "vQKiZBBZf", "marks": 80},
    {"name": "pMg", "marks": 63},
    {"name": "Rk23GA", "marks": 97},
    {"name": "n", "marks": 47},
    {"name": "eys2B", "marks": 85},
]
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 0.0 if norm_a == 0 or norm_b == 0 else np.dot(a, b) / (norm_a * norm_b)


@app.get("/")
async def root():
    return "Hello"


@app.get("/api")
async def get_names(request: Request):
    query_string = str(request.url.query).split("&")
    mark = []
    for query in query_string:
        name = query.split("=")[1]
        for entry in marks:
            if entry["name"] == name:
                mark.append(entry["marks"])
    return {"marks": mark}


@app.get("/api2")
async def get_names(request: Request):
    result = {"students": []}

    if request.url.query == "":
        for stu in student:
            result["students"].append(stu)

        return result

    result = {"students": []}

    query_string = str(request.url.query).split("&")
    classes = []
    for query in query_string:
        classes.append(query.split("=")[1])
    for entry in student:
        if entry["class"] in classes:
            result["students"].append(entry)
    return result


@app.post("/similarity")
async def get_similar_docs(request: Request, request_body: Dict):
    try:
        docs: List[str] = request_body.get("docs")
        query: str = request_body.get("query")

        if not docs or not query:
            raise HTTPException(
                status_code=400, detail="Missing 'docs' or 'query' in request body"
            )

        input_texts = [query] + docs

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
        }
        data = {"model": "text-embedding-3-small", "input": input_texts}
        embeddings_response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
            headers=headers,
            json=data,
        )
        embeddings_response.raise_for_status()
        embeddings_data = embeddings_response.json()
        query_embedding = embeddings_data["data"][0]["embedding"]
        doc_embedding = [emb["embedding"] for emb in embeddings_data["data"][1:]]
        similarities = [
            (i, cosine_similarity(query_embedding, doc_embedding[i]), docs[i])
            for i in range(len(docs))
        ]
        ranked_docs = sorted(similarities, key=lambda x: x[1], reverse=True)
        top_matches = [doc for _, _, doc in ranked_docs[: min(3, len(ranked_docs))]]
        return {"matches": top_matches}
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500, detail=f"Error communicating with AI Proxy: {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occured: {e}")


def get_ticket_status(ticket_id: int):
    return {"ticket_id": ticket_id}


def schedule_meeting(date: str, time: str, meeting_room: str):
    return {"date": date, "time": time, "meeting_room": meeting_room}


def get_expense_balance(employee_id: int):
    return {"employee_id": employee_id}


def calculate_performance_bonus(employee_id: int, current_year: int):
    return {"employee_id": employee_id, "current_year": current_year}


def report_office_issue(issue_code: int, department: str):
    return {"issue_code": issue_code, "department": department}


@app.get("/execute")
async def execute_query(q: str):
    try:
        query = q.lower()
        pattern_debug_info = {}

        # Ticket status pattern
        if re.search(r"ticket.*?\d+", query):
            ticket_id = int(re.search(r"ticket.*?(\d+)", query).group(1))
            return {
                "name": "get_ticket_status",
                "arguments": json.dumps({"ticket_id": ticket_id}),
            }
        pattern_debug_info["ticket_status"] = (
            re.search(r"ticket.*?\d+", query) is not None
        )

        # Meeting scheduling pattern
        if re.search(
            r"schedule.?\d{4}-\d{2}-\d{2}.?\d{2}:\d{2}.*?room", query, re.IGNORECASE
        ):
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
            time_match = re.search(r"(\d{2}:\d{2})", query)
            room_match = re.search(r"room\s*([A-Za-z0-9]+)", query, re.IGNORECASE)
            if date_match and time_match and room_match:
                return {
                    "name": "schedule_meeting",
                    "arguments": json.dumps(
                        {
                            "date": date_match.group(1),
                            "time": time_match.group(1),
                            "meeting_room": f"Room {room_match.group(1).capitalize()}",
                        }
                    ),
                }
        pattern_debug_info["meeting_scheduling"] = (
            re.search(
                r"schedule.?\d{4}-\d{2}-\d{2}.?\d{2}:\d{2}.*?room", query, re.IGNORECASE
            )
            is not None
        )

        # Expense balance pattern
        if re.search(r"expense", query):
            emp_match = re.search(r"employee\s*(\d+)", query, re.IGNORECASE)
            if emp_match:
                return {
                    "name": "get_expense_balance",
                    "arguments": json.dumps({"employee_id": int(emp_match.group(1))}),
                }
        pattern_debug_info["expense_balance"] = re.search(r"expense", query) is not None

        # Performance bonus pattern
        if re.search(r"bonus", query, re.IGNORECASE):
            emp_match = re.search(r"emp(?:loyee)?\s*(\d+)", query, re.IGNORECASE)
            year_match = re.search(r"\b(2024|2025)\b", query)
            if emp_match and year_match:
                return {
                    "name": "calculate_performance_bonus",
                    "arguments": json.dumps(
                        {
                            "employee_id": int(emp_match.group(1)),
                            "current_year": int(year_match.group(1)),
                        }
                    ),
                }
        pattern_debug_info["performance_bonus"] = (
            re.search(r"bonus", query, re.IGNORECASE) is not None
        )

        # Office issue pattern
        if re.search(r"(office issue|report issue)", query, re.IGNORECASE):
            code_match = re.search(r"(issue|number|code)\s*(\d+)", query, re.IGNORECASE)
            dept_match = re.search(
                r"(in|for the)\s+(\w+)(\s+department)?", query, re.IGNORECASE
            )
            if code_match and dept_match:
                return {
                    "name": "report_office_issue",
                    "arguments": json.dumps(
                        {
                            "issue_code": int(code_match.group(2)),
                            "department": dept_match.group(2).capitalize(),
                        }
                    ),
                }
        pattern_debug_info["office_issue"] = (
            re.search(r"(office issue|report issue)", query, re.IGNORECASE) is not None
        )

        raise HTTPException(status_code=400, detail=f"Could not parse query: {q}")

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse query: {q}. Error: {str(e)}. Pattern matches: {pattern_debug_info}",
        )

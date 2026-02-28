import sys
import os

modules = [
    "aiofiles",
    "anyio",
    "fastapi",
    "httpcore",
    "httpx",
    "jinja2",
    "markupsafe",
    "numpy",
    "orjson",
    "pydantic",
    "pydantic_core",
    "pydub",
    "starlette",
    "uvicorn",
    "gradio.utils",
    "gradio.helpers",
    "gradio.layouts",
    "gradio.components",
    "gradio.blocks"
]

for mod in modules:
    print(f"Importing {mod}...", end=" ")
    try:
        __import__(mod)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
    except BaseException as e:
        print(f"CRITICAL: {e}")

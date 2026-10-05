import json
from typing import Callable

from pydantic import BaseModel, ValidationError

MAX_RESULT_CHARS = 3500  # keeps each step small: free-tier limits count every resent token
CLOSE_TAG = "</tool_result>"


def wrap_untrusted(text: str) -> str:
    """Mark tool output as data. Remove any fake closing tag a document might contain."""
    text = text.replace(CLOSE_TAG, "[removed]")[:MAX_RESULT_CHARS]
    return f'<tool_result untrusted="true">\n{text}\n{CLOSE_TAG}'


class Tool:
    """A function the model may request, with typed input and output."""

    def __init__(self, name: str, description: str,
                 args_model: type[BaseModel], fn: Callable[..., BaseModel]):
        self.name, self.description, self.args_model, self.fn = name, description, args_model, fn

    def schema(self) -> dict:
        """The tool description in the format the chat API expects."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.args_model.model_json_schema()}}

    def run(self, raw_arguments: str) -> str:
        """Validate the model's arguments, run the tool, return wrapped text. Never raises."""
        try:
            args = self.args_model.model_validate_json(raw_arguments or "{}")
            output = self.fn(**args.model_dump())
            return wrap_untrusted(output.model_dump_json())
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            return f"ERROR: invalid arguments for {self.name}: {problems}"
        except (ValueError, KeyError) as e:
            return f"ERROR: {self.name} failed: {e}"


class ToolRegistry:
    """The ONLY set of tools the agent can use. Anything else is refused."""

    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, raw_arguments: str) -> str:
        if name not in self.tools:
            return f"ERROR: unknown tool '{name}'. Available tools: {sorted(self.tools)}"
        return self.tools[name].run(raw_arguments)

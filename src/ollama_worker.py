import logging
from typing import Literal

from ollama import AsyncClient
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("OllamaWorker")

RESPONSE_TYPES = ("SUMMARIZE", "GENERATE", "PLAN", "MEMORY")
ResponseType = Literal["SUMMARIZE", "GENERATE", "PLAN", "MEMORY"]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class SummarizeOutput(BaseModel):
    summary: str
    key_points: list[str]


class GenerateOutput(BaseModel):
    content: str
    style: str


class PlanStep(BaseModel):
    step: int
    action: str
    rationale: str


class PlanOutput(BaseModel):
    goal: str
    steps: list[PlanStep]


class MemoryItem(BaseModel):
    key: str
    value: str
    tags: list[str]


class MemoryOutput(BaseModel):
    memories: list[MemoryItem]


class AgentResponse(BaseModel):
    success: bool
    task: ResponseType
    model: str
    data: SummarizeOutput | GenerateOutput | PlanOutput | MemoryOutput | None = None
    error: str | None = None
    raw_content: str | None = None


class OllamaWorker:
    """Structured Ollama worker for task-specific agent responses.

    Supported tasks are SUMMARIZE, GENERATE, PLAN, and MEMORY. The worker keeps
    the model warm in RAM by sending keep_alive=True on every request.
    """

    def __init__(self, model_name: str = "gemma4:e2b"):
        self.model_name = model_name
        self.client = AsyncClient()
        self._schema_map = {
            "SUMMARIZE": SummarizeOutput,
            "GENERATE": GenerateOutput,
            "PLAN": PlanOutput,
            "MEMORY": MemoryOutput,
        }

    def _build_summarize_agent_messages(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, str]]:
        """Build prompts for the summarize agent.

        The summarize agent returns a concise summary and key bullet points.
        """
        system_message = {
            "role": "system",
            "content": (
                "You are the SUMMARIZE agent. Return valid JSON only with keys: "
                "summary (string) and key_points (array of strings)."
            ),
        }
        return [system_message, *[message.model_dump() for message in messages]]

    def _build_generate_agent_messages(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, str]]:
        """Build prompts for the generate agent.

        The generate agent produces content and labels the writing style used.
        """
        system_message = {
            "role": "system",
            "content": (
                "You are the GENERATE agent. Return valid JSON only with keys: "
                "content (string) and style (string)."
            ),
        }
        return [system_message, *[message.model_dump() for message in messages]]

    def _build_plan_agent_messages(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, str]]:
        """Build prompts for the planning agent.

        The planning agent returns a goal and a sequence of actionable steps.
        """
        system_message = {
            "role": "system",
            "content": (
                "You are the PLAN agent. Return valid JSON only with keys: "
                "goal (string) and steps (array of objects with step, action, rationale)."
            ),
        }
        return [system_message, *[message.model_dump() for message in messages]]

    def _build_memory_agent_messages(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, str]]:
        """Build prompts for the memory agent.

        The memory agent extracts durable memory items with key, value, and tags.
        """
        system_message = {
            "role": "system",
            "content": (
                "You are the MEMORY agent. Return valid JSON only with key memories "
                "(array of objects with key, value, tags)."
            ),
        }
        return [system_message, *[message.model_dump() for message in messages]]

    def _build_agent_messages(
        self, messages: list[ChatMessage], response_type: ResponseType
    ) -> list[dict[str, str]]:
        """Route input messages through the correct task-specific agent builder."""
        if response_type == "SUMMARIZE":
            return self._build_summarize_agent_messages(messages)
        if response_type == "GENERATE":
            return self._build_generate_agent_messages(messages)
        if response_type == "PLAN":
            return self._build_plan_agent_messages(messages)
        return self._build_memory_agent_messages(messages)

    async def generate_response(
        self, messages: list[dict[str, str]], response_type: ResponseType = "GENERATE"
    ) -> AgentResponse:
        """Generate structured task output from Ollama.

        Args:
            messages: Chat messages, each with role and content.
            response_type: Task to run. Must be one of RESPONSE_TYPES.

        Returns:
            AgentResponse containing typed task data when parsing succeeds.
            On failures, returns success=False with error details.
        """
        try:
            if response_type not in RESPONSE_TYPES:
                logger.warning(
                    "Invalid response type received | task=%s", response_type
                )
                return AgentResponse(
                    success=False,
                    task="GENERATE",
                    model=self.model_name,
                    error=f"Invalid response type: {response_type}",
                )

            logger.info(
                "Starting Ollama task request | task=%s model=%s message_count=%s",
                response_type,
                self.model_name,
                len(messages),
            )

            validated_messages = [
                ChatMessage.model_validate(message) for message in messages
            ]
            task_messages = self._build_agent_messages(
                validated_messages, response_type
            )
            schema_model = self._schema_map[response_type]

            logger.info("Dispatching chat request to Ollama | task=%s", response_type)
            response = await self.client.chat(
                model=self.model_name,
                messages=task_messages,
                format=schema_model.model_json_schema(),
                keep_alive=True,
            )

            raw_content = (
                response.message.content or "" if response and response.message else ""
            )
            logger.info(
                "Received raw chat response | task=%s content_length=%s",
                response_type,
                len(raw_content or ""),
            )

            parsed_data = schema_model.model_validate_json(raw_content)
            logger.info(
                "Parsed structured response successfully | task=%s", response_type
            )
            return AgentResponse(
                success=True,
                task=response_type,
                model=self.model_name,
                data=parsed_data,
                raw_content=raw_content,
            )
        except ValidationError as exc:
            logger.error(
                "Validation error in Ollama worker | task=%s error=%s",
                response_type,
                exc,
            )
            return AgentResponse(
                success=False,
                task=response_type,
                model=self.model_name,
                error=f"Validation error: {exc}",
            )
        except Exception as e:
            logger.exception(
                "Unhandled Ollama generation failure | task=%s", response_type
            )
            return AgentResponse(
                success=False,
                task=response_type,
                model=self.model_name,
                error=f"Error generating response from Ollama: {e}",
            )

import logging
import re

from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError

from app.config import Settings
from app.utils.errors import AppError

logger = logging.getLogger(__name__)


class LLMResult:
    def __init__(self, text: str, tokens_used: int | None = None) -> None:
        self.text = text
        self.tokens_used = tokens_used


class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None
        if (
            settings.use_openai
            and settings.llm_provider.lower() == "openai"
            and settings.llm_api_key
        ):
            self._client = OpenAI(
                api_key=settings.llm_api_key,
                timeout=settings.request_timeout_seconds,
            )

    @property
    def uses_external_api(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str, context: str, question: str) -> LLMResult:
        if self._client:
            return self._generate_with_openai(prompt)
        logger.warning("Using grounded extractive fallback for LLM response.")
        return LLMResult(text=self._extractive_answer(context, question), tokens_used=None)

    def _generate_with_openai(self, prompt: str) -> LLMResult:
        try:
            response = self._client.chat.completions.create(
                model=self.settings.openai_model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": "You answer only from retrieved knowledge-base context.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else None
            return LLMResult(text=text.strip(), tokens_used=tokens)
        except AuthenticationError as exc:
            raise AppError("Invalid LLM API key", status_code=401) from exc
        except RateLimitError as exc:
            raise AppError("LLM rate limit exceeded", status_code=429) from exc
        except APIConnectionError as exc:
            raise AppError("LLM request timeout or connection error", status_code=504) from exc
        except Exception as exc:
            raise AppError("LLM provider failed", status_code=502) from exc

    def _extractive_answer(self, context: str, question: str) -> str:
        context = self._content_only(context)
        question_terms = {
            term.lower()
            for term in re.findall(r"[A-Za-z][A-Za-z0-9-]+", question)
            if len(term) > 2
        }
        sentences = re.split(r"(?<=[.!?])\s+", context)
        ranked = sorted(
            sentences,
            key=lambda sentence: len(
                question_terms.intersection({word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9-]+", sentence)})
            ),
            reverse=True,
        )
        answer = " ".join(sentence.strip() for sentence in ranked[:3] if sentence.strip())
        return answer or "I could not find enough information in the knowledge base to answer this question."

    def _content_only(self, context: str) -> str:
        lines = []
        for line in context.splitlines():
            if line.startswith("Source:") or line.startswith("["):
                continue
            if line.startswith("Content:"):
                lines.append(line.replace("Content:", "", 1).strip())
            else:
                lines.append(line)
        return "\n".join(lines)

"""
Intelligent Scrape Orchestrator
--------------------------------
Async, fault-tolerant orchestrator that drives a size-targeted web-scraping
pipeline.  All non-critical work (DB writes, website-stat upserts, memory
compaction, session-total updates) is offloaded to the background Scheduler so
the hot scraping loop is never blocked by I/O.
"""

import asyncio
import re
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

from src.utils.db_manager import scrapes_db_manager
from src.utils.file_manager import EventLog, ManageFiles, SessionFileManager
from src.utils.utility import get_time, get_file_size_mb
from src.utils.bg_workers import Scheduler
from src.ollama_worker import OllamaWorker
from src.web.scraper import search_and_scrape_pages

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SCRAPES_DIR = DATA_DIR / "scrapes"
MAX_MEMORY_POINTS = 20
MAX_QUERY_RETRIES = 3
LOG_ID = "orchestrator"
LOC = "src/orchestrator.py"


# ---------------------------------------------------------------------------
# Helper – slugify
# ---------------------------------------------------------------------------
def _slugify(text: str) -> str:
    """Return a filesystem-safe slug from *text*."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:60]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class IntelligentScrapeOrchestrator:
    """
    Drives the full scraping lifecycle:

    1. Collect user topic + target MB.
    2. Generate & validate search queries via LLM.
    3. Scrape pages until the session folder reaches *target_mb*.
    4. Persist everything to SQLite; offload DB / memory work to Scheduler.
    """

    def __init__(self) -> None:
        self.logger = EventLog()
        self.memory_manager = ManageFiles("memory")
        self.file_ops = SessionFileManager()
        self.llm = OllamaWorker()
        self.scheduler = Scheduler(workers=3)

        # Session state (populated in start_session)
        self.session_id: Optional[int] = None
        self.topic: str = ""
        self.target_mb: float = 0.0
        self.session_folder: Optional[Path] = None
        self.scraped_websites: set[str] = set()  # dedup guard for this session
        self.memory_points: list[str] = []

    # ------------------------------------------------------------------
    # DB table setup
    # ------------------------------------------------------------------
    def setup_tables(self) -> None:
        """Create the three required SQLite tables if they don't exist yet."""
        scrapes_db_manager.create_table(
            "session",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "prompt": "TEXT NOT NULL",
                "created_at": "TEXT NOT NULL",
                "size_corpus": "REAL NOT NULL",
                "total_websites": "INTEGER DEFAULT 0",
            },
        )

        scrapes_db_manager.create_table(
            "scrapes",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "website": "TEXT NOT NULL",
                "source_url": "TEXT NOT NULL",
                "datetime": "TEXT NOT NULL",
                "query": "TEXT NOT NULL",
                "session": "INTEGER NOT NULL",
            },
            foreign_keys=[
                {
                    "column": "session",
                    "references_table": "session",
                    "references_column": "id",
                }
            ],
        )

        scrapes_db_manager.create_table(
            "website",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "website": "TEXT NOT NULL",
                "num_pages": "INTEGER NOT NULL DEFAULT 0",
                "session": "INTEGER NOT NULL",
            },
            indexes=[["website", "session"]],
            foreign_keys=[
                {
                    "column": "session",
                    "references_table": "session",
                    "references_column": "id",
                }
            ],
        )

        self.logger.info(LOG_ID, LOC, "DB tables ensured.")

    # ------------------------------------------------------------------
    # Session startup
    # ------------------------------------------------------------------
    async def start_session(self, topic: str, target_mb: float) -> None:
        """
        Bootstrap a new scraping session:
        initialise folders, DB row, memory, then drive the scraping loop.
        """
        self.topic = topic
        self.target_mb = target_mb
        self.scraped_websites = set()

        await self.scheduler.start()

        timestamp_str = get_time().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{_slugify(topic)}_{timestamp_str}"
        self.session_folder = SCRAPES_DIR / folder_name
        self.file_ops.ensure_dir(self.session_folder)

        # Insert session row (blocking – we need the ID immediately)
        result = scrapes_db_manager.insert(
            "session",
            {
                "prompt": topic,
                "created_at": get_time().isoformat(),
                "size_corpus": target_mb,
                "total_websites": 0,
            },
        )
        self.session_id = result.get("data", {}).get("id") if result else None
        if not self.session_id:
            self.logger.error(LOG_ID, LOC, "Failed to create session row – aborting.")
            return

        self.logger.info(
            LOG_ID,
            LOC,
            f"Session {self.session_id} started | topic='{topic}' | target={target_mb} MB",
        )

        # Memory bootstrap – fire-and-forget via scheduler
        self.load_memory_points()
        await self.append_memory_point(f"Topic: {topic}")
        await self.append_memory_point(f"Target corpus size: {target_mb} MB")

        # Main pipeline
        try:
            queries = await self.generate_queries()
            if not queries:
                self.logger.error(LOG_ID, LOC, "No valid queries generated – aborting.")
                return

            await self.scrape_until_target(queries)

        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(LOG_ID, LOC, f"Unhandled error in session: {exc}")

        finally:
            final_mb = self.calculate_session_size_mb()
            self.logger.info(
                LOG_ID,
                LOC,
                f"Session {self.session_id} complete | "
                f"corpus={final_mb:.2f} MB | "
                f"unique_sites={len(self.scraped_websites)}",
            )
            await self.scheduler.shutdown()

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------
    def load_memory_points(self) -> None:
        """Read bullet points from memory.md into *self.memory_points*."""
        raw = self.memory_manager.read()
        if raw.startswith("Error:"):
            self.memory_points = []
            return
        self.memory_points = [
            line.lstrip("- ").strip()
            for line in raw.splitlines()
            if line.strip().startswith("-")
        ]
        self.logger.info(
            LOG_ID, LOC, f"Loaded {len(self.memory_points)} memory points."
        )

    async def append_memory_point(self, point: str) -> None:
        """Add a bullet point to memory then schedule a compaction check."""
        self.memory_points.append(point)
        # Persist immediately (cheap write)
        self._flush_memory()
        # Compaction is non-critical – offload to scheduler
        await self.scheduler.schedule(
            self._bg_compact_memory_if_needed,
            params={},
        )

    def _flush_memory(self) -> None:
        """Write current memory_points list back to memory.md."""
        content = "\n".join(f"- {p}" for p in self.memory_points)
        self.memory_manager.write(content)

    def _bg_compact_memory_if_needed(self) -> None:
        """
        Background task: if memory exceeds MAX_MEMORY_POINTS, ask the LLM to
        compress it (MEMORY response_type) without information loss.

        Runs inside a Scheduler worker thread (sync context). Since OllamaWorker
        uses AsyncClient, we spin up a fresh event loop for this one call so the
        worker thread stays non-blocking relative to the main loop.
        """
        if len(self.memory_points) <= MAX_MEMORY_POINTS:
            return

        self.logger.info(LOG_ID, LOC, "Compacting memory via LLM…")
        try:
            raw_points = "\n".join(f"- {p}" for p in self.memory_points)
            messages = [
                {
                    "role": "system",
                    "content": "You are the MEMORY agent managing a scraping session.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Compress the following memory bullet points into at most "
                        f"{MAX_MEMORY_POINTS} items with NO information loss. "
                        "Each item's 'key' should be a short label, 'value' the fact, "
                        "and 'tags' relevant keywords.\n\n"
                        f"{raw_points}"
                    ),
                },
            ]

            # Run async LLM call in a dedicated event loop for this worker thread
            agent_response = asyncio.run(
                self.llm.generate_response(messages, response_type="MEMORY")
            )

            if not agent_response.success or not agent_response.data:
                self.logger.warning(
                    LOG_ID,
                    LOC,
                    f"Memory compaction LLM call failed: {agent_response.error}",
                )
                return

            # MemoryOutput.memories is list[MemoryItem]; flatten back to bullet strings
            compressed_points = [
                f"{item.key}: {item.value}"
                for item in agent_response.data.memories  # type: ignore[union-attr]
            ]
            if compressed_points:
                self.memory_points = compressed_points
                self._flush_memory()
                self.logger.info(
                    LOG_ID,
                    LOC,
                    f"Memory compacted to {len(self.memory_points)} points.",
                )

        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(LOG_ID, LOC, f"Memory compaction failed: {exc}")

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------
    def build_prompt_with_memory(self, system_content: str) -> list[dict]:
        """
        Return a messages list with the system prompt followed by the current
        memory context injected as an assistant turn.
        """
        memory_block = "\n".join(f"- {p}" for p in self.memory_points)
        messages = [
            {"role": "system", "content": system_content},
            {
                "role": "assistant",
                "content": f"[Memory context]\n{memory_block}",
            },
        ]
        return messages

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------
    async def generate_queries(self) -> list[str]:
        """
        Ask the LLM (PLAN) to produce search queries for the topic.
        Each step in the returned PlanOutput is used as one search query.
        Validates each query; regenerates off-topic ones up to MAX_QUERY_RETRIES.
        Returns the final validated list.
        """
        self.logger.info(LOG_ID, LOC, f"Generating queries for topic: '{self.topic}'")

        messages = self.build_prompt_with_memory(
            "You are a research planner for a web-scraping corpus pipeline."
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Topic: {self.topic}\n"
                    f"Target corpus size: {self.target_mb} MB\n"
                    "Create a search plan with 8 focused steps. "
                    "Each step's 'action' field should be a specific web search query. "
                    "The 'rationale' should explain why that query serves the corpus goal."
                ),
            }
        )

        raw_queries: list[str] = []
        for attempt in range(1, MAX_QUERY_RETRIES + 1):
            try:
                agent_response = await self.llm.generate_response(
                    messages, response_type="PLAN"
                )
                if not agent_response.success or not agent_response.data:
                    self.logger.warning(
                        LOG_ID,
                        LOC,
                        f"Query generation attempt {attempt} returned no data: "
                        f"{agent_response.error}",
                    )
                    await asyncio.sleep(2**attempt)
                    continue

                # PlanOutput.steps is list[PlanStep]; each .action is a search query
                raw_queries = [
                    step.action
                    for step in agent_response.data.steps  # type: ignore[union-attr]
                    if step.action.strip()
                ]
                self.logger.info(
                    LOG_ID,
                    LOC,
                    f"Raw queries attempt {attempt}: {len(raw_queries)} received.",
                )
                if raw_queries:
                    break

            except Exception as exc:  # pylint: disable=broad-except
                self.logger.error(
                    LOG_ID, LOC, f"Query generation attempt {attempt} failed: {exc}"
                )
                await asyncio.sleep(2**attempt)

        validated_queries: list[str] = []
        for query in raw_queries:
            if await self.validate_query(query):
                validated_queries.append(query)
            else:
                replacement = await self._regenerate_single_query(query)
                if replacement:
                    validated_queries.append(replacement)

        self.logger.info(LOG_ID, LOC, f"Final validated queries: {validated_queries}")
        return validated_queries

    async def validate_query(self, query: str) -> bool:
        """
        Ask the LLM (GENERATE) whether *query* is tightly aligned to the topic.
        Reads the boolean from GenerateOutput.content which should contain 'yes' or 'no'.
        Returns True if aligned, False otherwise.
        """
        messages = self.build_prompt_with_memory("You are a strict query validator.")
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Topic: {self.topic}\n"
                    f"Query: {query}\n"
                    "Is this query tightly aligned to the topic? "
                    "Set content to 'yes' or 'no' and style to 'validation'."
                ),
            }
        )
        try:
            agent_response = await self.llm.generate_response(
                messages, response_type="GENERATE"
            )
            if not agent_response.success or not agent_response.data:
                self.logger.warning(
                    LOG_ID, LOC, f"Validator got no data – defaulting PASS"
                )
                return True  # fail-open

            # GenerateOutput.content holds the 'yes'/'no' answer
            answer = agent_response.data.content.strip().lower()  # type: ignore[union-attr]
            is_valid = answer.startswith("yes")
            self.logger.info(
                LOG_ID,
                LOC,
                f"Validated query='{query}' → {'PASS' if is_valid else 'FAIL'} (raw='{answer}')",
            )
            return is_valid

        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning(
                LOG_ID, LOC, f"Query validation error: {exc} – defaulting PASS"
            )
            return True  # fail-open: don't block pipeline on validator error

    async def _regenerate_single_query(self, bad_query: str) -> Optional[str]:
        """Ask LLM (GENERATE) to suggest one replacement query for a rejected one."""
        messages = self.build_prompt_with_memory("You are a search query specialist.")
        messages.append(
            {
                "role": "user",
                "content": (
                    f"The query '{bad_query}' was rejected as off-topic.\n"
                    f"Topic: {self.topic}\n"
                    "Suggest exactly ONE replacement search query in the content field. "
                    "Set style to 'replacement'."
                ),
            }
        )
        try:
            agent_response = await self.llm.generate_response(
                messages, response_type="GENERATE"
            )
            if not agent_response.success or not agent_response.data:
                return None

            replacement = agent_response.data.content.strip()  # type: ignore[union-attr]
            if replacement and await self.validate_query(replacement):
                return replacement

        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning(
                LOG_ID, LOC, f"Replacement query generation failed: {exc}"
            )
        return None

    # ------------------------------------------------------------------
    # Main scraping loop
    # ------------------------------------------------------------------
    async def scrape_until_target(self, queries: list[str]) -> None:
        """
        Keep scraping rounds until the session folder size reaches *target_mb*.
        Each round iterates over all queries; new rounds recycle the same queries
        (the scraper deduplicates at the URL level within the session).
        """
        round_number = 0
        while True:
            current_mb = self.calculate_session_size_mb()
            self.logger.info(
                LOG_ID,
                LOC,
                f"Scrape round {round_number} | corpus={current_mb:.2f}/{self.target_mb} MB",
            )

            if current_mb >= self.target_mb:
                self.logger.info(LOG_ID, LOC, "Target corpus size reached – stopping.")
                break

            for query in queries:
                current_mb = self.calculate_session_size_mb()
                if current_mb >= self.target_mb:
                    break

                self.logger.info(LOG_ID, LOC, f"Running query: '{query}'")
                try:
                    async for page in search_and_scrape_pages([query]):
                        current_mb = self.calculate_session_size_mb()
                        if current_mb >= self.target_mb:
                            break
                        await self._handle_scraped_page(page, query)
                except Exception as exc:  # pylint: disable=broad-except
                    self.logger.error(
                        LOG_ID, LOC, f"Scrape error for query '{query}': {exc}"
                    )

            round_number += 1
            # Safety: if we scraped a full round but gained nothing, abort
            new_mb = self.calculate_session_size_mb()
            if new_mb == current_mb and round_number > 1:
                self.logger.warning(
                    LOG_ID, LOC, "No new content after full round – stopping."
                )
                break

    async def _handle_scraped_page(self, page: dict, query: str) -> None:
        """Process one scraped page result from the scraper pipeline."""
        if not page.get("success"):
            self.logger.warning(
                LOG_ID, LOC, f"Failed page: {page.get('url', 'unknown')}"
            )
            return

        source_url: str = page.get("url", "")
        content: str = page.get("content") or ""
        title: str = page.get("title") or ""

        if not source_url or not content:
            return

        website_host = self.extract_website(source_url)
        if website_host in self.scraped_websites:
            self.logger.info(LOG_ID, LOC, f"Skipping duplicate website: {website_host}")
            return

        # --- Write markdown file (blocking – size needs to be recalculated after) ---
        md_path = self.save_page_markdown(title or website_host, content, source_url)
        if not md_path:
            return

        # Mark website as seen immediately in-memory
        self.scraped_websites.add(website_host)

        scrape_datetime = get_time().isoformat()

        # --- Offload all DB / stat work to background scheduler ---

        # 1. Insert scrape row
        await self.scheduler.schedule(
            self._bg_insert_scrape_row,
            params={
                "website": website_host,
                "source_url": source_url,
                "scrape_datetime": scrape_datetime,
                "query": query,
                "session_id": self.session_id,
            },
        )

        # 2. Upsert website stats
        await self.scheduler.schedule(
            self._bg_upsert_website_stats,
            params={
                "website_host": website_host,
                "session_id": self.session_id,
            },
        )

        # 3. Update session total_websites count
        await self.scheduler.schedule(
            self._bg_update_session_total,
            params={
                "session_id": self.session_id,
                "total": len(self.scraped_websites),
            },
        )

        # 4. Append a memory note (also async – goes via append_memory_point which
        #    schedules compaction as a further background task)
        await self.append_memory_point(f"Scraped: {website_host} | {title[:60]}")

        self.logger.info(
            LOG_ID,
            LOC,
            f"Saved page: {website_host} | size={self.calculate_session_size_mb():.2f} MB",
        )

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    def save_page_markdown(
        self, title_or_host: str, content: str, source_url: str
    ) -> Optional[Path]:
        """
        Persist a scraped page to a timestamped markdown file in the session folder.
        Returns the file path on success, None on failure.
        """
        try:
            timestamp = get_time().strftime("%Y%m%d_%H%M%S")
            return self.file_ops.write_scraped_markdown(
                session_folder=self.session_folder,  # type: ignore[arg-type]
                title_or_host=title_or_host,
                source_url=source_url,
                content=content,
                timestamp=timestamp,
                slugify_fn=_slugify,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(
                LOG_ID, LOC, f"Failed to save markdown for '{title_or_host}': {exc}"
            )
            return None

    def extract_website(self, url: str) -> str:
        """Return the normalised hostname (e.g. 'en.wikipedia.org') from a URL."""
        try:
            return urlparse(url).netloc.lower().lstrip("www.")
        except Exception:  # pylint: disable=broad-except
            return url

    def calculate_session_size_mb(self) -> float:
        """Sum *get_file_size_mb* across all .md files in the session folder."""
        if not self.session_folder or not self.session_folder.exists():
            return 0.0
        total = 0.0
        for md_file in self.session_folder.glob("*.md"):
            try:
                total += get_file_size_mb(md_file)
            except FileNotFoundError:
                pass
        return round(total, 3)

    # ------------------------------------------------------------------
    # Background DB tasks (called by Scheduler workers)
    # These must be plain sync functions – Scheduler handles threading.
    # ------------------------------------------------------------------

    def _bg_insert_scrape_row(
        self,
        website: str,
        source_url: str,
        scrape_datetime: str,
        query: str,
        session_id: int,
    ) -> None:
        """Background: insert one row into the scrapes table."""
        try:
            scrapes_db_manager.insert(
                "scrapes",
                {
                    "website": website,
                    "source_url": source_url,
                    "datetime": scrape_datetime,
                    "query": query,
                    "session": session_id,
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(LOG_ID, LOC, f"BG insert_scrape_row failed: {exc}")

    def _bg_upsert_website_stats(self, website_host: str, session_id: int) -> None:
        """
        Background: upsert the website table row for *website_host*,
        incrementing num_pages by 1.
        """
        try:
            existing = scrapes_db_manager.fetch_one(
                "website",
                where={"website": website_host, "session": session_id},
            )
            if existing and existing.get("data"):
                current_pages = existing["data"].get("num_pages", 0)
                scrapes_db_manager.update(
                    "website",
                    data={"num_pages": current_pages + 1},
                    where={"website": website_host, "session": session_id},
                )
            else:
                scrapes_db_manager.insert(
                    "website",
                    {
                        "website": website_host,
                        "num_pages": 1,
                        "session": session_id,
                    },
                )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(LOG_ID, LOC, f"BG upsert_website_stats failed: {exc}")

    def _bg_update_session_total(self, session_id: int, total: int) -> None:
        """Background: update session.total_websites to reflect current unique count."""
        try:
            scrapes_db_manager.update(
                "session",
                data={"total_websites": total},
                where={"id": session_id},
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(LOG_ID, LOC, f"BG update_session_total failed: {exc}")

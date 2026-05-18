import asyncio

from src.orchestrator import IntelligentScrapeOrchestrator


async def main() -> None:
    print("=== Intelligent Scraper ===")
    topic = input("Enter topic/genre: ").strip()
    while not topic:
        topic = input("Topic cannot be empty. Enter topic: ").strip()

    target_mb_str = input("Enter target corpus size in MB (e.g. 10.5): ").strip()
    try:
        target_mb = float(target_mb_str)
        assert target_mb > 0
    except (ValueError, AssertionError):
        print("Invalid size. Defaulting to 10 MB.")
        target_mb = 10.0

    orchestrator = IntelligentScrapeOrchestrator()
    orchestrator.setup_tables()
    await orchestrator.start_session(topic, target_mb)


if __name__ == "__main__":
    asyncio.run(main())

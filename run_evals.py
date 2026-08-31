import asyncio
from evals.ablation import CONFIGS
from evals.runner import run

async def main():
    for name, config in CONFIGS:
        print(f"Running eval for: {name}")
        await run(config, split="dev")

if __name__ == "__main__":
    asyncio.run(main())

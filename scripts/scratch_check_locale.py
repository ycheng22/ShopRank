import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    pool = await asyncpg.create_pool(os.environ['DATABASE_URL'])
    rows = await pool.fetch("SELECT locale, count(1) FROM queries GROUP BY locale")
    print(rows)
    await pool.close()

if __name__ == '__main__':
    asyncio.run(main())

#!/usr/bin/env python3
import pika
import sys
import aio_pika

async def test_connection():
    try:
        # Try with SSL connection
        connection_url = "amqps://guest:guest@0.tcp.ngrok.io:15433/"
        # Or with SSL parameters
        connection = await aio_pika.connect_robust(
            "amqp://guest:guest@0.tcp.ngrok.io:15433/",
            ssl=True,
            timeout=10
        )
        print("Connection established successfully!")
        # Rest of your code...
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
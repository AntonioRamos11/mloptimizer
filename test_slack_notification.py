#!/usr/bin/env python3
"""Test script to verify Slack notification works"""

import os
import aiohttp
import asyncio

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

async def test_slack_notification():
    """Send a test message to Slack"""
    message = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🧪 MLOptimizer Test Notification"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "This is a test message to verify the Slack integration is working correctly!"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": "*Status:*\nTest Passed"
                    },
                    {
                        "type": "mrkdwn",
                        "text": "*Time:*\n" + str(asyncio.get_event_loop().time())
                    }
                ]
            }
        ]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(SLACK_WEBHOOK_URL, json=message) as response:
            if response.status == 200:
                print("✅ Slack notification test successful!")
                print("   Message sent to:", SLACK_WEBHOOK_URL)
            else:
                print("❌ Slack notification test failed!")
                print("   Status:", response.status)
                text = await response.text()
                print("   Response:", text)

if __name__ == "__main__":
    print("Testing Slack notification...")
    print(f"Webhook URL: {SLACK_WEBHOOK_URL}")
    print()
    asyncio.run(test_slack_notification())

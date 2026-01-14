# Requires: pip install discord.py
import discord
import os
from .queue_manager import queue

class RalphClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        # Don't respond to ourselves
        if message.author == self.user:
            return

        if message.content.startswith('Hey Ralph') or message.content.startswith('hey ralph'):
            task = message.content.replace('Hey Ralph', '').replace('hey ralph', '').strip()
            if task:
                queue.add_task(f"Discord:{message.author}", task)
                await message.channel.send(f"✅ Task queued: {task}")

def run_bot(token: str):
    intents = discord.Intents.default()
    intents.message_content = True
    client = RalphClient(intents=intents)
    client.run(token)

if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if token:
        run_bot(token)
    else:
        print("Error: DISCORD_TOKEN not found.")

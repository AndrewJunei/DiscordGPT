import discord
import csv
import json

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

client = discord.Client(intents=intents)

channel_id = 123
write_path = 'channels/name.csv'
num_collected, after_id = 0, 0

async def get_names():
    server_id = 123
    server = client.get_guild(server_id)

    with open('data/usernames.txt', 'r') as f:
        names = f.read()

    members = json.loads(names)

    for member in server.members:
        members[str(member.id)] = member.display_name

    with open('data/usernames.txt','w') as f:
        f.write(json.dumps(members, indent=4))

async def get_history(channel):
    global num_collected, after_id
    rows = []
    after = await channel.fetch_message(after_id)

    async for msg in channel.history(limit=100, after=after):
        after_id = msg.id
        timestamp = msg.created_at
        author = msg.author.id
        content = msg.content

        reactions = {}
        for r in msg.reactions:
            reactions[r.emoji] = r.count

        attachments = [a.url for a in msg.attachments]
        mentions = [m.id for m in msg.mentions]

        rows.append([timestamp, author, content, reactions, attachments, mentions])
        num_collected += 1
    
    with open(write_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    
    with open('log.txt', 'w') as f:
        f.writelines([str(num_collected), '\n', str(after_id)])

@client.event
async def on_ready():
    global num_collected, after_id
    print(f'Logged in as {client.user}')
    print('Collecting data...')

    channel = client.get_channel(channel_id)

    with open('log.txt', 'r') as f:
        log = f.readlines()
        num_collected = int(log[0])
        after_id = int(log[1])

    # await get_names()

    while True:
        await get_history(channel)

@client.event
async def on_message(message):
    if client.user in message.mentions:
        await message.reply(f'Collected {num_collected} messages...')


client.run('token')
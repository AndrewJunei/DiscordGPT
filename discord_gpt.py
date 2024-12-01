import discord
import torch
import json
import re
import random
from GPT2.model import GPT2LMHeadModel
from GPT2.config_med import GPT2Config
from GPT2.encoder import get_encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = get_encoder()
config = GPT2Config()
model = GPT2LMHeadModel(config)
model.load_state_dict(torch.load('gpt_checkpoint_8.pth', map_location=device))
model.eval()
print('Model loaded...')

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)

with open('users.json', 'r') as f:
    user_dict = json.load(f)
reverse_user_dict = {v:k for k,v in user_dict.items()}

with open('usernames.txt', 'r') as f:
    names = json.load(f)

with open('full_text.txt', 'r', encoding='utf-8') as f:
    chars = f.read()
    chars = list(set(chars))


@torch.no_grad()
def generate_msg(model, device, context): # generates one message
    end = False
    temperature = 0.8
    prev = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0)
    output = prev
    past = None

    while not end:
        logits, past = model(prev, past=past)
        logits = logits[:, -1] / temperature 
        log_probs = torch.nn.functional.softmax(logits, dim=-1)
        prev = torch.multinomial(log_probs, num_samples=1)
        output = torch.cat((output, prev), dim=1)

        if output[0].tolist()[-3:] == [1279, 437, 29]:
            end = True
    return output

def clean_text(text):
    url_pattern = r'https?://\S+'
    emote_pattern = r'<:[^>]+>'
    other_emote_pattern = r'<a:[^>]+>'
    at_pattern = r'<@(\d+)>'
    hash_pattern = r'<#\d+>'
    role_pattern = r'<@&\d+>'

    # remove URLs and emotes, etc
    text = re.sub(url_pattern, '', text)
    text = re.sub(emote_pattern, '', text)
    text = re.sub(other_emote_pattern, '', text)
    text = re.sub(hash_pattern, '', text)
    text = re.sub(role_pattern, '', text)

    # replace <@! with <@ (old @ user format)
    text = text.replace("<@!", "<@")

    # function to replace mentions with "@userint" using the user_dict
    def replace_with_username(match):
        at_string = match.group(0)  # Get the full matched string (e.g., "<@123>")
        return f"{user_dict.get(at_string, '')}" # default value on right

    # replace mentions
    text = re.sub(at_pattern, replace_with_username, text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = ''.join(c for c in text if c in chars)
    text = text.strip()
    return text

def send_msg(context_txt):
    global history
    ctxlen = 128

    context = enc.encode(context_txt)
    history.extend(context)
    history = history[-ctxlen:]

    generated = generate_msg(model, device, history)[0].tolist() # generates until <end>
    new_tokens = generated[len(history):] 
    new_msg = enc.decode(new_tokens[:-3])

    user_pattern = r"user(\d+): "
    match = re.search(user_pattern, new_msg)

    if match:
        username = match.group()
        new_msg = new_msg.replace(username, '', 1) 
    else:
        username = 'blank'

    at_pattern = r"@user(\d+)"
    def replace_with_username(match):
        at_string = match.group(0) 
        return f"{reverse_user_dict.get(at_string, '')}"
    new_msg = re.sub(at_pattern, replace_with_username, new_msg)

    history.extend(new_tokens + [220, 198]) # add space + \n
    history = history[-ctxlen:]
    return new_msg, username


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

history = []
busy = False

@client.event
async def on_message(message):
    global history, busy
    if client.user in message.mentions and message.content != '' and not busy:
        busy = True
        real_at = '<@bot_ID>'
        content = message.content.replace(real_at, '')
        input_txt = clean_text(content)

        if not input_txt.isspace() and input_txt != '':
            async with message.channel.typing():
                auth_id = message.author.id

                user_int = '<' + str(auth_id) + '>' # user that sent the message
                user_str = user_dict.get(user_int, random.choice(list(user_dict.values())))

                context_txt = user_str + input_txt + ' <end> \n' # + custom_user

                new_msg, username = send_msg(context_txt)
                if new_msg in ['lol', 'Lol', 'Lmao', 'lmao']: 
                    new_msg, username = send_msg(context_txt) # re-generate once if this happens

            user_int = reverse_user_dict.get(username, "<1>")
            name = names.get(user_int[1:-1], 'random')
            print(name, ':')
            print(new_msg, '\n')
            await message.reply(new_msg)
        busy = False


client.run('token')
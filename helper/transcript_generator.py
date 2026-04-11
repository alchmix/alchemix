import os
import asyncio
import openai
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import tiktoken

# Load API credentials
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)


# Load tokenizer for gpt-4o
encoding = tiktoken.encoding_for_model("gpt-4o")

def token_count(text):
    return len(encoding.encode(text))
async def punctuate_text(text_block):
    """Use GPT to restore punctuation without paraphrasing."""
    prompt = f"""
You are a transcript editor. Your task is to fix punctuation, grammar, and sentence flow ONLY.

- DO NOT change any wording.
- DO NOT paraphrase or add new content.
- DO NOT remove or alter any [CLIP:...] markers.

Example:
"she said she was fine but i knew she was lying"
→ "She said she was fine, but I knew she was lying."

Fix this:
{text_block}

Return only the fixed version.
"""
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
import re

def split_safe(text, max_tokens=5000):
    """Split long text into clean chunks based on punctuation and token limits."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_token_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_token_len = token_count(sentence)

        # If sentence is too big alone, just force split it safely
        if sentence_token_len >= max_tokens:
            midpoint = len(sentence) // 2
            part1 = sentence[:midpoint].rsplit(' ', 1)[0]
            part2 = sentence[len(part1):]
            chunks.extend(split_by_punctuation_and_token_limit(part1, max_tokens))
            chunks.extend(split_by_punctuation_and_token_limit(part2, max_tokens))
            continue

        if current_token_len + sentence_token_len <= max_tokens:
            current_chunk.append(sentence)
            current_token_len += sentence_token_len
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_token_len = sentence_token_len

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

async def paraphrase_segment(text, context, niche, pbar):
    """Paraphrase narration segments while preserving clip markers."""
    if not text.strip():
        return text

    if niche == 'history':
        prompt = f"""
PARAPHRASE THIS NARRATIVE:
- Keep the factual integrity and chronological flow intact
- Vary phrasing and sentence structure while keeping the same impact
- Match the tone of a documentary or historical exposé
- Emphasize suspense, context, and deeper implications
- Use cinematic and descriptive language, suitable for voice-over

CONTEXT:
{context}

SEGMENT_TEXT:
{text}

Provide ONLY the paraphrased version below. No extra commentary.
"""
    
    elif niche == 'philosophy':
        prompt = f"""
PARAPHRASE THIS PHILOSOPHICAL THOUGHT:
- Preserve the original meaning and logical structure
- Use reflective, contemplative language
- Vary sentence flow while keeping the same message
- Make it sound intellectually engaging and introspective
- Suitable for philosophical narration or thoughtful commentary

CONTEXT:
{context}

SEGMENT_TEXT:
{text}

Return ONLY the paraphrased version below. No extra commentary.
"""

    else:
        prompt = f"""
PARAPHRASE TASK:
- Keep same meaning and dramatic impact
- Maintain same emotional tone
- Use different wording/sentence structures
- Keep conversational, gossipy style
- Ensure complete sentences with proper grammar
- Avoid sentence fragments or incomplete transitions

CONTEXT:
{context}

SEGMENT_TEXT:
{text}

Return ONLY the paraphrased version below. No extra commentary.
"""

    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    pbar.update(1)
    return response.choices[0].message.content.strip()

# def parse_script(script_content):
#     """Improved parser for multi-line clip markers"""
#     elements = []
#     current_clip = []
#     current_narration = []
#     in_clip = False
    
#     for line in script_content.split('\n'):
#         line = line.strip()
#         if line.startswith('[CLIP:'):
#             in_clip = True
#             if current_narration:
#                 elements.append(('narration', '\n'.join(current_narration)))
#                 current_narration = []
#             current_clip.append(line)
#         elif in_clip and line == ']':
#             current_clip.append(line)
#             elements.append(('clip', '\n'.join(current_clip)))
#             current_clip = []
#             in_clip = False
#         elif in_clip:
#             current_clip.append(line)
#         else:
#             current_narration.append(line)
    
#     if current_narration:
#         elements.append(('narration', '\n'.join(current_narration)))
    
#     return elements
def split_into_three_chunks(text):
    """Split text into 3 equal-length parts (not by sentence). Avoid splitting short texts."""
    words = text.split()
    total = len(words)

    # 🚨 Safeguard: Don't split if less than 100 words
    if total < 100:
        return [text]

    part_size = total // 3
    remainder = total % 3
    split_points = [
        part_size + (1 if remainder > 0 else 0),
        2 * part_size + (1 if remainder > 1 else 0)
    ]
    chunk1 = ' '.join(words[:split_points[0]])
    chunk2 = ' '.join(words[split_points[0]:split_points[1]])
    chunk3 = ' '.join(words[split_points[1]:])
    return [chunk1, chunk2, chunk3]

# def parse_script(script_content):
#     """Parse narration and clip sections. Handle [CLIP: ... ] even if start and end are on same line."""
#     elements = []
#     current_narration = []
#     in_clip = False
#     current_clip_lines = []

#     # Flatten all lines for flexible processing
#     lines = script_content.split('\n')
#     i = 0
#     while i < len(lines):
#         line = lines[i].strip()

#         if '[CLIP:' in line:
#             # First, flush narration if any
#             if current_narration:
#                 narration_text = ' '.join(current_narration)
#                 chunks = split_into_three_chunks(narration_text)
#                 for chunk in chunks:
#                     elements.append(('narration', chunk))
#                 current_narration = []

#             in_clip = True
#             current_clip_lines = [line]

#             # If this line also has closing `]`, it's a one-liner clip
#             if ']' in line and line.index(']') > line.index('[CLIP:'):
#                 elements.append(('clip', '\n'.join(current_clip_lines)))
#                 in_clip = False
#                 current_clip_lines = []
#             else:
#                 # Search until we find a line containing closing `]`
#                 i += 1
#                 while i < len(lines):
#                     next_line = lines[i].strip()
#                     current_clip_lines.append(next_line)
#                     if ']' in next_line:
#                         elements.append(('clip', '\n'.join(current_clip_lines)))
#                         in_clip = False
#                         current_clip_lines = []
#                         break
#                     i += 1
#         else:
#             if not in_clip:
#                 current_narration.append(line)
#         i += 1

#     # Final catch for narration
#     if current_narration:
#         narration_text = ' '.join(current_narration)
#         chunks = split_into_three_chunks(narration_text)
#         for chunk in chunks:
#             elements.append(('narration', chunk))

#     return elements

def token_split_narration(text, max_tokens=2000):
    """Split a long narration block into multiple chunks based on token count."""
    words = text.strip().split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if token_count(' '.join(current_chunk)) >= max_tokens:
            chunks.append(' '.join(current_chunk).strip())
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())

    return chunks


# def parse_script(script_content):
#     """Strictly parse narration and [CLIP:...] sections, ending exactly at `]`."""
#     elements = []
#     current_narration = []
#     current_clip_lines = []

#     lines = script_content.split('\n')
#     i = 0

#     while i < len(lines):
#         line = lines[i].strip()

#         if '[CLIP:' in line:
#             # Flush existing narration
#             if current_narration:
#                 narration_text = ' '.join(current_narration).strip()
#                 if narration_text:
#                     elements.append(('narration', narration_text))
#                 current_narration = []

#             current_clip_lines = [line]

#             # If ] is on the same line (one-liner or mid-line clip end)
#             if ']' in line and line.index(']') > line.index('[CLIP:'):
#                 clip_part, post_clip = line.split(']', 1)
#                 elements.append(('clip', clip_part.strip() + ']'))
#                 if post_clip.strip():
#                     current_narration.append(post_clip.strip())
#             else:
#                 # Multiline clip
#                 i += 1
#                 while i < len(lines):
#                     next_line = lines[i].strip()

#                     if ']' in next_line:
#                         before, after = next_line.split(']', 1)
#                         current_clip_lines.append(before.strip() + ']')
#                         elements.append(('clip', '\n'.join(current_clip_lines).strip()))
#                         if after.strip():
#                             current_narration.append(after.strip())
#                         break
#                     else:
#                         current_clip_lines.append(next_line)
#                     i += 1
#         else:
#             current_narration.append(line)
#         i += 1

#     # Final flush
#     if current_narration:
#         narration_text = ' '.join(current_narration).strip()
#         if narration_text:
#             elements.append(('narration', narration_text))

#     return elements

def parse_script(script_content):
    """Strictly parse narration and [CLIP:...] sections, ending exactly at `]`, with token-based narration chunking."""
    elements = []
    current_narration = []
    current_clip_lines = []

    lines = script_content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if '[CLIP:' in line:
            # Flush existing narration
            if current_narration:
                narration_text = ' '.join(current_narration).strip()
                if narration_text:
                    for chunk in token_split_narration(narration_text, max_tokens=2000):
                        elements.append(('narration', chunk))
                current_narration = []

            current_clip_lines = [line]

            # If ] is on the same line (one-liner or mid-line clip end)
            if ']' in line and line.index(']') > line.index('[CLIP:'):
                clip_part, post_clip = line.split(']', 1)
                elements.append(('clip', clip_part.strip() + ']'))
                if post_clip.strip():
                    current_narration.append(post_clip.strip())
            else:
                # Multiline clip
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()

                    if ']' in next_line:
                        before, after = next_line.split(']', 1)
                        current_clip_lines.append(before.strip() + ']')
                        elements.append(('clip', '\n'.join(current_clip_lines).strip()))
                        if after.strip():
                            current_narration.append(after.strip())
                        break
                    else:
                        current_clip_lines.append(next_line)
                    i += 1
        else:
            current_narration.append(line)
        i += 1

    # Final flush
    if current_narration:
        narration_text = ' '.join(current_narration).strip()
        if narration_text:
            for chunk in token_split_narration(narration_text, max_tokens=2000):
                elements.append(('narration', chunk))

    return elements


import math
import re

# def get_tail_words(text, min_words=20):
#     """Extract the last part of the text with at least `min_words`, ending at a sentence boundary going backward."""
#     if not text:
#         return ''
    
#     words = text.strip().split()
#     total_words = len(words)
#     if total_words <= min_words:
#         return text.strip()

#     tail_words = []
#     count = 0

#     # Start from the end and move backwards
#     for word in reversed(words):
#         tail_words.append(word)
#         count += 1

#         # Stop if we reach punctuation *and* have collected at least min_words
#         if count >= min_words and re.search(r'[.!?]$', word):
#             break

#     # Reverse the list to restore original order
#     tail_words.reverse()
#     final = ' '.join(tail_words).strip()

#     # Fallback punctuation if none found
#     if not final.endswith(('.', '!', '?')):
#         final += '.'

#     return final

def get_tail_words(text, min_words=20):
    """Get the last few words from a previous segment to maintain flow"""
    return ' '.join(text.strip().split()[-min_words:]) if text else ''



# async def process_transcript_with_sections(original_script, niche, intro_ratio=0.3):
#     """Dynamically split script into intro/body with context-aware paraphrasing."""
#     elements = parse_script(original_script)  # already split into 3 narration chunks per block

#     # Identify narration chunks
#     narration_indices = [idx for idx, (etype, content) in enumerate(elements) if etype == 'narration']
#     total_narrations = len(narration_indices)

#     if total_narrations == 0:
#         return {
#             "intro": "",
#             "body": "",
#             "full_script": ""
#         }

#     # Compute dynamic split (minimum 1 element in both parts)
#     if total_narrations == 1:
#         intro_count = 1
#         body_count = 0
#     elif total_narrations == 2:
#         intro_count = 1
#         body_count = 1
#     else:
#         intro_count = int(round(total_narrations * intro_ratio))
#         intro_count = max(1, min(intro_count, total_narrations - 1))  # force at least 1 for both parts
#         body_count = total_narrations - intro_count

#     intro_narration_idxs = narration_indices[:intro_count]
#     body_narration_idxs = narration_indices[intro_count:]

#     # Prepare paraphrasing with context
#     paraphrase_tasks = [(idx, elements[idx][1]) for idx in narration_indices]

#     from tqdm.asyncio import tqdm
#     with tqdm(total=len(paraphrase_tasks), desc="Paraphrasing") as pbar:
#         prev_text = ""
#         tasks = []
#         for idx, content in paraphrase_tasks:
#             context_tail = get_tail_words(prev_text,20) if prev_text else ''
#             text = content
#             context_tail=context_tail

#             tasks.append((idx, paraphrase_segment(text,context_tail, niche, pbar)))
#             prev_text = content

#         results = await asyncio.gather(*[task[1] for task in tasks])
#         paraphrased = {paraphrase_tasks[i][0]: results[i] for i in range(len(results))}

#     # Rebuild full script sections
#     intro, body = [], []

#     for idx, (etype, content) in enumerate(elements):
#         if etype == 'narration':
#             rewritten = paraphrased.get(idx, content)
#             if idx in intro_narration_idxs:
#                 intro.append(rewritten)
#             else:
#                 body.append(rewritten)
#         else:
#             # non-narration content (e.g., clips) goes to the section based on nearest narration
#             nearest_intro = narration_indices[intro_count - 1] if intro_narration_idxs else -1
#             if idx <= nearest_intro:
#                 intro.append(content)
#             else:
#                 body.append(content)

#     return {
#         "intro": '\n'.join(intro),
#         "body": '\n'.join(body),
#         "full_script": '\n'.join(intro + body)
#     }
async def process_transcript_with_sections(original_script, niche, intro_ratio=0.3):
    """Dynamically split script into intro/body with context-aware paraphrasing. 
    If niche is 'original', skip paraphrasing and use the original narration blocks."""
    
    elements = parse_script(original_script)

    # Identify narration chunks
    narration_indices = [idx for idx, (etype, content) in enumerate(elements) if etype == 'narration']
    total_narrations = len(narration_indices)

    if total_narrations == 0:
        return {
            "intro": "",
            "body": "",
            "full_script": ""
        }

    # Compute dynamic split (minimum 1 element in both parts)
    if total_narrations == 1:
        intro_count = 1
        body_count = 0
    elif total_narrations == 2:
        intro_count = 1
        body_count = 1
    else:
        intro_count = int(round(total_narrations * intro_ratio))
        intro_count = max(1, min(intro_count, total_narrations - 1))  # force at least 1 for both parts
        body_count = total_narrations - intro_count

    intro_narration_idxs = narration_indices[:intro_count]
    body_narration_idxs = narration_indices[intro_count:]

    paraphrased = {}

    if niche == 'original':
        # Skip paraphrasing, use original narration blocks
        for idx in narration_indices:
            paraphrased[idx] = elements[idx][1]
    else:
        # Perform paraphrasing with context
        paraphrase_tasks = [(idx, elements[idx][1]) for idx in narration_indices]
        with tqdm(total=len(paraphrase_tasks), desc="Paraphrasing") as pbar:
            prev_text = ""
            tasks = []
            for idx, content in paraphrase_tasks:
                context_tail = get_tail_words(prev_text, 20) if prev_text else ''
                tasks.append((idx, paraphrase_segment(content, context_tail, niche, pbar)))
                prev_text = content

            results = await asyncio.gather(*[task[1] for task in tasks])
            paraphrased = {paraphrase_tasks[i][0]: results[i] for i in range(len(results))}

    # Rebuild full script sections
    intro, body = [], []

    for idx, (etype, content) in enumerate(elements):
        if etype == 'narration':
            rewritten = paraphrased.get(idx, content)
            if idx in intro_narration_idxs:
                intro.append(rewritten)
            else:
                body.append(rewritten)
        else:
            # Non-narration content (e.g., clips) goes to section based on nearest narration
            nearest_intro = narration_indices[intro_count - 1] if intro_narration_idxs else -1
            if idx <= nearest_intro:
                intro.append(content)
            else:
                body.append(content)

    return {
        "intro": '\n'.join(intro),
        "body": '\n'.join(body),
        "full_script": '\n'.join(intro + body)
    }


async def generate_viral_title(script_content):
    """Generate YouTube title from script content"""
    prompt = f"""
    Create a viral YouTube title based on this script:
    {script_content[:200]}... [truncated]
    
    Requirements:
    - Under 70 characters with emojis
    - Include key names/controversies
    - Use CAPS for emphasis
    - Create urgency/drama
    - SEO-friendly keywords
    
    Examples:
    "🔥 BILL MAHER SHOCKS FANS! Defends Trump in WILD Rant - You Won't Believe What He Said!"
    "💥 MAHER vs WOKE CULTURE! Explosive Clip Leaks - The Internet is FUMING!"
    """
    
    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()

# async def generate_section_titles(script_content):
#     """Generate engaging section titles for the body content"""
#     prompt = f"""
#     Create 3-5 engaging section titles for a video script about Bill Maher's controversial opinions.
#     The titles should be catchy, provocative, and relevant to the content.
    
#     Script content:
#     {script_content[:3000]}... [truncated]
    
#     Requirements:
#     - Each title should be 2-5 words
#     - Use provocative language
#     - Focus on controversial topics mentioned
#     - Emphasize drama and viewer engagement
    
#     Return ONLY the section titles, one per line, without numbering or additional text.
#     """
    
#     response = await openai_client.chat.completions.create(
#         model="gpt-4-turbo",
#         messages=[{"role": "user", "content": prompt}]
#     )
    
#     return response.choices[0].message.content.strip().split('\n')

async def full_processing_pipeline(transcript, niche, intro_ratio=3):
    """End-to-end processing with intro/body split and title generation"""
    # Process transcript with sections
    fixed_transcript = re.sub(r'(?<!\n)(\[CLIP:)', r'\n\1', transcript)
    processed_sections = await process_transcript_with_sections(fixed_transcript,niche, intro_ratio)
    
    
    print(processed_sections)
    
    # Generate title
    title = await generate_viral_title(processed_sections["full_script"])
    
    # Generate section titles for body
    # section_titles = await generate_section_titles(processed_sections["body"])
    
    return {
        "title": title,
        "intro": processed_sections["intro"],
        "body": processed_sections["body"],
        "full_script": processed_sections["full_script"],
        "original_clip_count": len([e for e in parse_script(transcript) if e[0] == 'clip'])
    }

# Example usage:
# transcript = """
  
# [CLIP:
# Sports by sex doesn't make sense. Yes, it does
# Because again, we haven't reinvented homo sapiens since crystal Pepsi came out
# I've spent three decades on TV mocking Republicans who said climate change was just a theory and now I got to deal with people who
# Say, you know, what else is just a theory by a but that can't happen overnight
# Because the people who aren't getting the food or the medicine they're the ones who are gonna suffer
# You don't see why people find this incredibly cruel
# You said earlier that we're treating it as if it's a it's an overweight person putting them on a crash diet
# oh, no, the federal government is essentially somebody who's 800 pounds and
# You cannot survive like this
# So yes
# There are some drastic measures that have to be taken and I will add this to you
# If you think it's bad now if we get into a real debt crisis
# It will be significantly worse for the American people and other people that we might help around the globe the best thing to do right now
# Getting quick. You're gonna have to shut some things down triage the area rebuild the stuff
# That is good that we all can agree with the other stuff. That's hyper political or makes no sense or fraudulent
# What about the Department of Education now? I I don't know that much about it, but I've never read good things
# Rahm Emanuel who I agree with almost everything here was a quote
# He said a third of eighth graders can't read and now he wants to close the department
# ]
# Alright, folks, Here we go.. Bill Maher, right?
# He's seriously stirring the pot these days, and it’s got everyone talking, remember, just hearing about it might actually, make your blood boil.
# The iconic comedian and HBO Show host has taken a bold stance that’s, honestly, caught everyone off guard—he’s defending Donald Trump!

# [CLIP:
# If I give you a billion dollars and you can't tell me what happened to it, that to me is
# wasteful.
# That means you are not responsible.
# But if you can't tell me where it went, then what am I supposed to think?
# An $850 billion budget to an organization that can't pass an audit and tell you where
# that money went, like, I think most people would consider that somewhere in the realm
# of waste, fraud, or abuse because they would wonder why that money isn't well accounted
# for.
# But I'm a human being who lives on the earth and can't figure out how $850 billion to a
# department means that the rank and file still have to be on food stamps.
# Like to me, that's fucking corruption.
# ]

# And, yes, that's the guy he’s mocked for years.
# Can you believe it? Maher’s stepping up, critiquing both sides of the political aisle without missing a beat.

# [CLIP:
# With communists, that human was no longer selfish.
# In America today, that human is no longer born male or female.
# And obesity is not something that affects health.
# You can be healthy at any size.
# Really, we voted on it.
# A formerly serious magazine last year
# published with a straight face an article called,
# Separating Sports by Sex Doesn't Make Sense.
# Yes, it does.
# Because again, we haven't reinvented homo sapien
# since crystal Pepsi came out.
# I've spent three decades on TV mocking Republicans
# who said climate change was just a theory.
# And now I've got to deal with people who say,
# you know what else is just a theory?
# ]

# In today's fiery political climate, his latest act, tackling hot-button issues like education and immigration, has left his audience divided—some are cheering, while others are, like, fuming mad!
# Wait a sec! If you enjoy content that challenges the norm and keeps it real, hit that subscribe button.
# Go on, I’ll wait... Done? Welcome aboard! Now, let’s jump back in.
# Bill Maher's Views
# Alright, let's dive into.. all the nitty-gritty details of Bill Maher’s unfiltered views, shall we?
# Recently, Maher took to his platform to lay bare his opinion on some of the most challenging societal issues of our time.
# And he doesn't hold back, folks.
# His candid commentary has been hitting hard on topics like the Department of Education, immigration policies out in California, and oh boy, those touchy gender discussions in sports.
# Now, let's start with Bill's take on education.
# I mean, how do I put this..?
# For decades, education has been this sacred untouchable realm, right?
# But Maher's calling it out, pointing and saying, "Look!
# Something's not adding up here." To be honest, his critique isn't just a shot in the dark—it’s based on figures that are hard to ignore.
# For example, a troubling report card shows that a third of our country’s eighth graders can’t read at grade level.
# So, Maher’s standing there, practically shouting, "What on earth is going on here?" I mean, seriously, have we been pouring billions into a system that's, like, getting worse every year?
# If you ask Bill, he would say the Department of Education feels more like a money pit than an education advocate, kind of like a black hole where tax dollars disappear.
# Immigration Policies
# Then, there’s Maher on immigration. Now, this is where it gets even spicier.
# You see, California’s lawmakers decided to extend government assistance to undocumented immigrants to buy houses.

# [CLIP:
# The California lawmakers just passed a law giving government assistance to undocumented
# immigrants to buy houses.
# That's kind of a different place than the Democratic Party used to be on immigration,
# is it not?
# Well, let me just say, immigration had always been a bipartisan issue.
# I refer you to the-
# But not free houses.
# Well, that's not free housing.
# It's the American dream, being available to more people.
# California's always in the lead.
# Maybe others will follow that lead, but that's up to those states.
# So you'd vote for this law?
# I'm not familiar with exactly what it is, but making the American dream of home ownership
# available to all people is something we have to do for people who are here now.
# This is for the undocumented.
# ]

# Wow, kind of unexpected, right? Now, I hear you, California’s always breaking new ground, setting trends for others to maybe follow, but Maher’s pondering, "Is this really where the Democratic party used to stand on immigration?" It’s like, been such a bipartisan issue for ages.
# Think about it, we’re all for extending the American Dream to more people, but shouldn't it be for, like, Americans?
# Maher’s skeptically raising an eyebrow and kind of wondering if maybe, just maybe, the terms are shifting a bit too far.
# For Bill—this doesn’t compute. He’s not saying don’t support people, but man, he’s asking the tough questions, just getting right to the core of it.
# Gender Discussions
# And then, there’s the conversation that leaves the genders discussing in sports up for debate.
# You know, there’s President Trump, issuing an executive order to keep boys from competing in girls’ sports.
# Some people cheered, others, not so much.
# But Maher, well, he’s chiming in with his usual dose of comedy and sharp critique.
# Instead of picking sides, guess what..?

# [CLIP:
# President Trump this week just did an executive order saying that boys can no longer play in girls' sports.
# That's it.
# That's it.
# Ladies, isn't it ironic that the biggest protector of women turned out to be the pussy grabber?
# Huh?
# Maybe the guy who grabs it knows how to protect it the best.
# Think about that.
# Think about that.
# ]

# Maher’s digging into the irony of it all.
# Trump's being seen as a protector of women, yet, his infamous past involvement in scandals seems contradictory.
# Provoking Thoughts
# But here’s what I love about Maher.
# It’s not just infuriation. It's not just comedy.
# He’s provoking genuine thought about societal norms.
# While, you know, letting everyone who listens in on the joke and saves their seat for the tweets that are sure to follow.
# Oh, and there's more! Maher’s recent jabs at progressives are yet another spectacle.
# This time, he’s calling out the indoctrination he sees in today's youth.
# He’s connecting dots from their support for Hamas, a rather contentious stance, to their anti-American sentiment.
# Believe it or not, Bill’s arguing for a return to patriotism in schools.
# He’s pushing for education that loves America, not teaching contempt for it.
# In one of his typical jabs, Maher even been caught highlighting Senate Majority Leader Chuck Schumer’s warning about anti-Semitism, a huge issue burning in today’s headlines.
# Schumer's statements about hostile chants and their implications about erasure resonate with Maher.
# It’s kind of nerve-wracking to see this unfold, I must say.
# Maher’s standing tall, side by side, amplifying the narrative Schumer's laid out.

# [CLIP:
# Under his administration, his next one, schools will teach students to love their country,
# not to hate their country, like they're taught now.
# I got to say, this one doesn't bother me so much, because I think that, I mean, this is
# what I see when I see kids demonstrating these progressives, demonstrating for Hamas, the
# most illiberal people in the world, that, oh, good, we're going to give America its
# comeuppance, asshole America.
# This is, they kind of have been indoctrinated this way.
# And Chuck Schumer made a speech this week.
# He said anti-Semitism is a five-alarm fire that must be extinguished.
# This is the highest-ranking Jewish politician we've ever had in this country.
# He talked about when Jewish people hear chants like, from the river to the sea, you understand
# that means wipe us out by any means necessary.
# Vulnerabilities in the party, James.
# Democratic Contradictions
# ]

# Amidst all of this, there’s this ongoing frustration within the Democratic ranks.
# I mean, how Democrats can get so wrapped up in their contradictions is baffling, right??
# Like, Maher’s seamless delivery of these cutting observations, it only just boggles the mind.
# Does all this ring a bell?
# The Democrats are often caught, sort of trapped in dealing with things in a roundabout, sometimes almost nonsensical way.
# Meanwhile, Maher—he's giving a masterclass on addressing issues head-on!
# It’s no wonder people are looking at him for this brutally genuine commentary.
# So, folks, I gotta ask..? What’s your take on this?
# Are you, like, flabbergasted by Maher's audacious approach or do you find some wisdom in his witticism?
# Drop your comments below and let me know what you think about Maher calling education a "black hole" or in questioning the American Dream's true beneficiaries.
# Can’t wait to see your thoughts on this!
# """


# result = asyncio.run(full_processing_pipeline(transcript, intro_length=3))
# print(f"Generated Title: {result['title']}")
# print(f"\nINTRO:\n{result['intro']}")
# print(f"\nBODY:\n{result['body']}")
# print(f"\nSection Titles: {result['section_titles']}")


# result = asyncio.run(full_processing_pipeline(transcript, emotion="controversial"))
# print(f"Generated Title: {result['title']}")
# print(f"Processed Script:\n{result['processed_script']}")


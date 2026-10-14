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
#     """Strictly parse narration and [CLIP:...] sections, ending exactly at `]`, with token-based narration chunking."""
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
#                     for chunk in token_split_narration(narration_text, max_tokens=2000):
#                         elements.append(('narration', chunk))
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
#             for chunk in token_split_narration(narration_text, max_tokens=2000):
#                 elements.append(('narration', chunk))

#     return elements

def split_narration_by_char_cap(text, max_chars=5000, min_tail=3000):
    """
    Split narration for TTS without breaking mid-word.
    - max_chars: 5000 (safe cap)
    - min_tail: 3000 — if peeling max_chars leaves less than this, rebalance evenly

    IRON RULE: if the entire block <= max_chars, it stays as ONE segment.
    Clips are handled separately in parse_script — never split across [CLIP:...].
    """
    text = text.strip()
    if not text:
        return []

    # Never split a block that already fits
    if len(text) <= max_chars:
        return [text]

    def word_boundary_cut(s, target):
        if len(s) <= target:
            return len(s)
        idx = min(target, len(s))
        while idx > 0 and not s[idx - 1].isspace():
            idx -= 1
        if idx == 0:
            idx = min(target, len(s))
            while idx < len(s) and not s[idx].isspace():
                idx += 1
        return idx

    def rebalance_evenly(block):
        """Split an oversized block into 2 equal parts at word boundaries."""
        block = block.strip()
        if len(block) <= max_chars:
            return [block]

        mid = len(block) // 2
        cut = word_boundary_cut(block, mid)
        if cut == 0 or cut >= len(block):
            cut = word_boundary_cut(block, max_chars)

        part1 = block[:cut].strip()
        part2 = block[cut:].strip()

        segments = []
        for part in (part1, part2):
            if not part:
                continue
            # Each part that fits stays as ONE segment — no further splitting
            if len(part) <= max_chars:
                segments.append(part)
            else:
                segments.extend(split_block(part))
        return segments

    def split_block(block):
        block = block.strip()
        if len(block) <= max_chars:
            return [block]

        cut = word_boundary_cut(block, max_chars)
        first = block[:cut].strip()
        rest = block[cut:].strip()

        if not rest:
            return [first] if first else []

        # Remainder big enough → keep the peel split
        if len(rest) >= min_tail:
            segments = [first] if first else []
            segments.extend(split_block(rest))
            return segments

        # Remainder too small → rebalance the whole block evenly instead
        return rebalance_evenly(block)

    return split_block(text)

def parse_script(script_content, narration_chunk_fn=None):
    """
    Parse narration and [CLIP:...] blocks.
    Clips are atomic — narration is only split inside a block via narration_chunk_fn,
    never across a [CLIP:...] marker.
    """
    if narration_chunk_fn is None:
        def narration_chunk_fn(text):
            return token_split_narration(text, max_tokens=2000)
    elements = []
    current_narration = []
    current_clip_lines = []
    lines = script_content.split('\n')
    i = 0
    def flush_narration():
        nonlocal current_narration
        if not current_narration:
            return
        narration_text = ' '.join(current_narration).strip()
        if narration_text:
            for chunk in narration_chunk_fn(narration_text):
                elements.append(('narration', chunk))
        current_narration = []
    while i < len(lines):
        line = lines[i].strip()
        if '[CLIP:' in line:
            flush_narration()
            current_clip_lines = [line]
            if ']' in line and line.index(']') > line.index('[CLIP:'):
                clip_part, post_clip = line.split(']', 1)
                elements.append(('clip', clip_part.strip() + ']'))
                if post_clip.strip():
                    current_narration.append(post_clip.strip())
            else:
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
                    current_clip_lines.append(next_line)
                    i += 1
        else:
            current_narration.append(line)
        i += 1
    flush_narration()
    return elements

import math
import re


def get_tail_words(text, min_words=20):
    """Get the last few words from a previous segment to maintain flow"""
    return ' '.join(text.strip().split()[-min_words:]) if text else ''


# async def process_transcript_with_sections(original_script, niche, intro_ratio=0.3):
#     """Dynamically split script into intro/body with context-aware paraphrasing. 
#     If niche is 'original', skip paraphrasing and use the original narration blocks."""
    
#     elements = parse_script(original_script)

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

#     paraphrased = {}

#     if niche == 'original':
#         # Skip paraphrasing, use original narration blocks
#         for idx in narration_indices:
#             paraphrased[idx] = elements[idx][1]
#     else:
#         # Perform paraphrasing with context
#         paraphrase_tasks = [(idx, elements[idx][1]) for idx in narration_indices]
#         with tqdm(total=len(paraphrase_tasks), desc="Paraphrasing") as pbar:
#             prev_text = ""
#             tasks = []
#             for idx, content in paraphrase_tasks:
#                 context_tail = get_tail_words(prev_text, 20) if prev_text else ''
#                 tasks.append((idx, paraphrase_segment(content, context_tail, niche, pbar)))
#                 prev_text = content

#             results = await asyncio.gather(*[task[1] for task in tasks])
#             paraphrased = {paraphrase_tasks[i][0]: results[i] for i in range(len(results))}

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
#             # Non-narration content (e.g., clips) goes to section based on nearest narration
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
async def process_transcript_with_sections(original_script, niche, intro_ratio=0.3, commentary_mode=None):
    """
    commentary_mode: 'paraphrase' | 'original' (only when niche == 'commentary')
    """
    is_commentary_original = niche == 'commentary' and commentary_mode == 'original'
    skip_paraphrase = niche == 'original' or is_commentary_original
    if is_commentary_original:
        def narration_chunk_fn(text):
            return split_narration_by_char_cap(text, max_chars=5000, min_tail=3000)
    else:
        def narration_chunk_fn(text):
            return token_split_narration(text, max_tokens=2000)
    elements = parse_script(original_script, narration_chunk_fn=narration_chunk_fn)
    narration_indices = [idx for idx, (etype, _) in enumerate(elements) if etype == 'narration']
    total_narrations = len(narration_indices)
    if total_narrations == 0:
        return {"intro": "", "body": "", "full_script": ""}
    if total_narrations == 1:
        intro_count = 1
    elif total_narrations == 2:
        intro_count = 1
    else:
        intro_count = int(round(total_narrations * intro_ratio))
        intro_count = max(1, min(intro_count, total_narrations - 1))
    intro_narration_idxs = set(narration_indices[:intro_count])
    paraphrased = {}
    if skip_paraphrase:
        for idx in narration_indices:
            paraphrased[idx] = elements[idx][1]
    else:
        paraphrase_tasks = [(idx, elements[idx][1]) for idx in narration_indices]
        with tqdm(total=len(paraphrase_tasks), desc="Paraphrasing") as pbar:
            prev_text = ""
            tasks = []
            for idx, content in paraphrase_tasks:
                context_tail = get_tail_words(prev_text, 20) if prev_text else ''
                tasks.append(paraphrase_segment(content, context_tail, niche, pbar))
                prev_text = content
            results = await asyncio.gather(*tasks)
            paraphrased = {paraphrase_tasks[i][0]: results[i] for i in range(len(results))}
    intro, body = [], []
    nearest_intro_clip_idx = narration_indices[intro_count - 1] if intro_narration_idxs else -1
    for idx, (etype, content) in enumerate(elements):
        if etype == 'narration':
            rewritten = paraphrased.get(idx, content)
            if idx in intro_narration_idxs:
                intro.append(rewritten)
            else:
                body.append(rewritten)
        else:
            if idx <= nearest_intro_clip_idx:
                intro.append(content)
            else:
                body.append(content)
    return {
        "intro": '\n'.join(intro),
        "body": '\n'.join(body),
        "full_script": '\n'.join(intro + body),
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


# async def full_processing_pipeline(transcript, niche, intro_ratio=3):
#     """End-to-end processing with intro/body split and title generation"""
#     # Process transcript with sections
#     fixed_transcript = re.sub(r'(?<!\n)(\[CLIP:)', r'\n\1', transcript)
#     processed_sections = await process_transcript_with_sections(fixed_transcript,niche, intro_ratio)
    
    
#     print(processed_sections)
    
#     # Generate title
#     title = await generate_viral_title(processed_sections["full_script"])
    
#     # Generate section titles for body
#     # section_titles = await generate_section_titles(processed_sections["body"])
    
#     return {
#         "title": title,
#         "intro": processed_sections["intro"],
#         "body": processed_sections["body"],
#         "full_script": processed_sections["full_script"],
#         "original_clip_count": len([e for e in parse_script(transcript) if e[0] == 'clip'])
#     }

async def full_processing_pipeline(transcript, niche, intro_ratio=3, commentary_mode=None):
    fixed_transcript = re.sub(r'(?<!\n)(\[CLIP:)', r'\n\1', transcript)
    processed_sections = await process_transcript_with_sections(
        fixed_transcript,
        niche,
        intro_ratio,
        commentary_mode=commentary_mode,
    )
    print(processed_sections)
    title = await generate_viral_title(processed_sections["full_script"])
    return {
        "title": title,
        "intro": processed_sections["intro"],
        "body": processed_sections["body"],
        "full_script": processed_sections["full_script"],
        "original_clip_count": len([e for e in parse_script(transcript) if e[0] == 'clip']),
    }


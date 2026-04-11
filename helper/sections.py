# import os
# import asyncio
# import openai
# from dotenv import load_dotenv
# from tqdm.asyncio import tqdm
# import logging
# import re
# import difflib
# import traceback
# from collections import Counter
# from typing import List, Optional

# # Load API credentials
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize OpenAI client
# openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

# # ========= OpenRouter client for AI segmentation =========
# OPENROUTER_API_KEY = "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"
# if not OPENROUTER_API_KEY or not OPENROUTER_API_KEY.strip():
#     raise SystemExit(
#         "ERROR: Set OPENROUTER_API_KEY in your environment (must start with 'sk-or-').\n"
#         "Windows PowerShell:  $env:OPENROUTER_API_KEY='sk-or-xxx'\n"
#         "Windows CMD:         set OPENROUTER_API_KEY=sk-or-xxx\n"
#         "macOS/Linux:         export OPENROUTER_API_KEY='sk-or-xxx'\n"
#     )

# openrouter_client = openai.OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=OPENROUTER_API_KEY,
# )

# # ========= AI Segmentation Constants =========
# CUT_TOK = "### CUT ###"
# _SEGMENT_SYSTEM = """
# You are a professional video editor working on a timeline.  
# Your task is to split the narration into visual segments so that each segment corresponds to exactly one image or clip that will appear on screen.  

# Think like you are matching visuals to voiceover — each segment is one visual beat.  
# When the voiceover shifts to a new subject, action, setting, or object, you cut.  

# Rules:
# - One segment = one image or shot.  
# - A segment must contain only one clear visual focus — one subject, action, object, or idea that can be illustrated without confusion.  
# - If a line contains multiple actions, subjects, or list items, split them individually.  
# - If a line expands on the same subject without introducing a new action or focus, keep it in the same segment.  
# - Segments may be slightly longer only when describing one single, continuous idea or static visual.  
# - Do NOT paraphrase, summarize, or add commentary. Preserve the original text exactly.  
# - Insert separator exactly as:
# ### CUT ###
# - Cut strictly as a video editor would — when you'd naturally place a new visual on the timeline.  

# Your goal:  
# Deliver a version of the narration that can be mapped 1:1 to visuals —  
# each segment producing one accurate, self-contained image representing that part of the story.
# """

# def clean_text(text):
#     """Clean the input text by removing unnecessary spaces and newlines."""
#     if not text or not isinstance(text, str):
#         return ""

#     # Remove extra spaces and newlines
#     cleaned_text = re.sub(r'\s+', ' ', text).strip()
#     return cleaned_text

# def _normalize_space(s: str) -> str:
#     return re.sub(r"\s+", " ", (s or "")).strip()

# def _word_count(s: str) -> int:
#     return len(re.findall(r"\w+", s or ""))

# def _sentences(text: str) -> List[str]:
#     # Lightweight sentence splitter; protects common abbreviations.
#     abbr = r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|e\.g|i\.e|etc)\."
#     safe = re.sub(rf"\b{abbr}", lambda m: m.group(0).replace(".", "§"), text or "")
#     parts = re.split(r"(?<=[\.!?])\s+(?=[A-Z0-9\"'(\[])", safe.strip())
#     parts = [p.replace("§", ".").strip() for p in parts if p.strip()]
#     return parts or ([text.strip()] if text and text.strip() else [])

# def _chunk_sentences(sentences: List[str], max_words: int, overlap_sentences: int) -> List[str]:
#     chunks: List[str] = []
#     i, n = 0, len(sentences)
#     while i < n:
#         words = 0
#         buff = []
#         start = i
#         while i < n:
#             s = sentences[i]
#             w = _word_count(s)
#             if buff and words + w > max_words:
#                 break
#             buff.append(s)
#             words += w
#             i += 1
#         chunks.append(" ".join(buff))
#         if i < n and overlap_sentences > 0:
#             i = max(i - overlap_sentences, start + 1)
#     return chunks

# def _llm_cut(client: openai.OpenAI, text: str, model: str, temperature: float) -> List[str]:
#     text = (text or "").strip()
#     if not text:
#         return []
    
#     try:
#         resp = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "system", "content": _SEGMENT_SYSTEM},
#                 {"role": "user", "content": text},
#             ],
#             temperature=temperature,
#         )
#         raw = (resp.choices[0].message.content or "").strip()
#         segs = [s.strip() for s in re.split(rf"\s*{re.escape(CUT_TOK)}\s*", raw) if s.strip()]
#         return segs if segs else [text]  # fallback if model didn't insert separators
#     except Exception as e:
#         logging.error(f"Error in AI segmentation: {e}")
#         return [text]  # fallback to original text

# def _trim_head_overlap(seg: str, overlap: str) -> str:
#     seg_n, ov_n = _normalize_space(seg), _normalize_space(overlap)
#     if ov_n and seg_n.startswith(ov_n) and len(seg_n) > len(ov_n):
#         return _normalize_space(seg_n[len(ov_n):])
#     return seg

# def _merge_tiny_segments(segments: List[str], min_chars: int) -> List[str]:
#     out: List[str] = []
#     for s in segments:
#         if not s.strip():
#             continue
#         if out and len(_normalize_space(s)) < min_chars:
#             out[-1] = _normalize_space(out[-1] + " " + s)
#         else:
#             out.append(_normalize_space(s))
#     return out

# def estimate_segments_atomic(
#     narration_text: str,
#     niche: str | None = None,
#     min_words: int = 0,        # kept for API compatibility; not used
#     absorb_threshold: int = 0  # kept for API compatibility; not used
# ) -> List[str]:
#     """
#     AI-powered scene-aware segmentation using OpenRouter.
#     Returns List[str] segments just like the original function.
#     """
#     narration_text = narration_text or ""
#     if not narration_text.strip():
#         return []

#     # Use AI segmentation with OpenRouter
#     sents = _sentences(narration_text)
#     chunks = _chunk_sentences(sents, max_words=220, overlap_sentences=1)

#     segments: List[str] = []
#     prev_tail: Optional[str] = None

#     def _tail_sentence_of(chunk_text: str) -> str:
#         ss = _sentences(chunk_text)
#         return ss[-1] if ss else ""

#     for idx, chunk in enumerate(chunks):
#         segs = _llm_cut(openrouter_client, chunk, model="deepseek/deepseek-chat-v3.1:free", temperature=0.3)

#         # Reconcile the overlap at the head of this chunk
#         if idx > 0 and prev_tail:
#             if segs:
#                 segs[0] = _trim_head_overlap(segs[0], prev_tail)
#             segs = [
#                 s for s in segs
#                 if _normalize_space(s) and _normalize_space(s) != _normalize_space(prev_tail)
#             ]

#         segments.extend(segs)
#         prev_tail = _tail_sentence_of(chunk) if 1 > 0 else None  # overlap_sentences = 1

#     # Merge tiny fragments and normalize
#     segments = _merge_tiny_segments(segments, min_chars=40)
#     segments = [_normalize_space(s) for s in segments if s.strip()]
    
#     return segments

# # Keep the original estimate_segments function for backward compatibility
# def estimate_segments(narration_text, niche, min_words=25, absorb_threshold=10):
#     """Original rule-based segmentation (fallback)"""
#     return estimate_segments_atomic(narration_text, niche, min_words, absorb_threshold)

# # ========= REST OF YOUR EXISTING CODE (unchanged) =========

# async def extract_character_names(text,niche):
#     """Extract potential celebrity names from text using NER via the API."""
#     try:
#         if (niche=='history'):
#             system_prompt = (
#              "-You are an expert in analyzing historical texts and identifying key entities.\n"
#              "-Extract ONLY the names of real historical figures, countries, empires, governments, or organizations mentioned in the text.\n"
#              "-Return each entity on a new line.\n"
#              "-Include ONLY the names—no extra commentary, numbers, or punctuation.\n"
#              "-For people, return full names if possible. If only a partial name is provided, use context to infer the full name.\n"
#              "-Do NOT include titles (e.g., 'King', 'President'), nicknames, or non-historical references.\n"
#              "-Do NOT include '[CLIP:...]' markers or any mention of video clips as characters.\n"
#              "-Focus only on historically relevant names that played significant roles in events.\n"
#              "-Exclude vague or general terms like 'the army', 'the people', or 'the leader'."
#         )
#         if(niche=='philosophy'):
#              print("In philosophy character")
#              system_prompt = (
#             "-You are a script analysis assistant for a video generation AI.\n"
#             "-From the given text, extract the most relevant terms that should be visually represented.\n"
#             "-Include:\n"
#             "  • Historical or philosophical figures (e.g., 'Socrates', 'Tesla')\n"
#             "  • Abstract ideas or themes (e.g., 'ego', 'truth', 'mirror effect')\n"
#             "  • Important action verbs that reflect the tone or movement (e.g., 'reject', 'reflect', 'challenge')\n"
#             "-Do NOT include punctuation,'[CLIP:...]' markers, or filler words.\n"
#             "-Keep everything in a clean, ready-to-use keyword list.\n"
#         )
#         else:
#            system_prompt = (
#     "-You are an expert at identifying named entities (people and organizations) in text.\n"
#     "-Extract ONLY the names of:\n"
#     "  • People: full names of real celebrities, politicians, public figures, or fictional characters.\n"
#     "  • Organizations: media outlets (e.g., CNN), companies, governments, political parties, NGOs, teams, institutions, platforms.\n"
#     "-Return each name on a new line.\n"
#     "-Include ONLY the names, with no additional text, numbers, or punctuation.\n"
#     "-For people: Provide full names only. If only a single name is provided, use context to guess the full name; if uncertain, omit.\n"
#     "-For organizations: Provide canonical names or widely used acronyms (e.g., CNN, NATO).\n"
#     "-Do not include titles, nicknames, handles, hashtags, or partial names.\n"
#     "-Do not include generic groups or concepts (e.g., the media, government, fans) unless they are recognized proper organizations.\n"
#     "-Normalize near-duplicates (e.g., US, U.S. → United States when referring to the government).\n"
#     "-List each unique entity once, in order of first appearance.\n"
#     "-Do Not include Clip as Character name.\n"
# )


#         response = await openai_client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": text}
#             ],
#         )

#         extracted_names = response.choices[0].message.content.strip().split('\n')
#         # Clean up and validate names
#         validated_names = []
#         for name in extracted_names:
#             name = name.strip()
#             # Only accept names with alphabetic characters
#             if name and re.match(r'^[A-Za-z\s\'-]+$', name):
#                 validated_names.append(name)
        
#         return validated_names
    
#     except Exception as e:
#         logging.error(f"Error extracting celebrity names: {e}")
#         return []

# async def process_segments_with_context(segments, celebrities):
#     """Process segments while maintaining context across multiple segments.
    
#     - Assigns [CLIP:...] segments the default character "Clip"
#     - Processes other segments while maintaining context
#     """
#     if not segments:
#         return []

#     try:
#         # Prepare context windows (up to 5 segments at a time)
#         context_windows = []
#         for i in range(len(segments)):
#             start = max(0, i-2)
#             end = min(len(segments), i+3)
#             context = " ".join(segments[start:end])

#             # Identify if the segment is a [CLIP:...] marker
#             if re.match(r"^\[CLIP:.*?\]$", segments[i]):
#                 character = "Clip"
#                 segment_text = segments[i]  # Ensure segment text is stored for Clip
#             else:
#                 character = "Narrator" if not celebrities else None  # To be assigned later
#                 segment_text = segments[i]

#             context_windows.append({
#                 "segment_index": i,
#                 "context": context,
#                 "segment": segment_text,  # Ensure segment text is present
#                 "character": character  # Pre-set for clips
#             })

#         # Process only the non-[CLIP:...] segments in batches
#         batch_size = 10
#         all_results = []

#         non_clip_batches = [
#             item for item in context_windows if item["character"] is None
#         ]

#         for i in range(0, len(non_clip_batches), batch_size):
#             batch = non_clip_batches[i:min(i+batch_size, len(non_clip_batches))]
#             batch_results = await analyze_batch_with_context(batch, celebrities)
#             all_results.extend(batch_results)

#         # Merge processed results with pre-set clip segments
#         for item in context_windows:
#             if item["character"] == "Clip":
#                 all_results.append({
#                     "segment_index": item["segment_index"],
#                     "character": "Clip",
#                     "text": item["segment"]  # ✅ Fix: Ensure Clip segments have "text"
#                 })

#         # Sort results by original segment index
#         all_results.sort(key=lambda x: x["segment_index"])

#         # ✅ Fix: Guarantee every item has "text"
#         return post_process_for_consistency([{
#             "character": item["character"],
#             "text": item.get("text", item.get("segment", "[MISSING TEXT]"))  # Prevents KeyError
#         } for item in all_results])

#     except Exception as e:
#         logging.error(f"Error processing segments with context: {traceback.format_exc()}")
#         return [{"character": "Clip" if re.match(r'^\[CLIP:.*?\]$', segment) else "Narrator", "text": segment} for segment in segments]



# async def analyze_batch_with_context(batch_contexts, celebrities):
#     """Assigns characters using GPT with strict Clip enforcement and smart fallback."""
#     try:
#         # Format input for GPT
#         context_texts = []
#         for i, item in enumerate(batch_contexts):
#             context_texts.append(
#                 f"SEGMENT {i+1} CONTEXT:\n{item['context']}\nSEGMENT TO ANALYZE:\n{item['segment']}"
#             )

#         combined_text = "\n\n".join(context_texts)
#         celebrity_list = ", ".join(celebrities)

   
        
#         system_prompt = (
#             f"You are an expert in analyzing text and determining who is being discussed.\n\n"
#             f"For each numbered segment, determine which person from this list is being discussed: {celebrity_list}.\n\n"
#             f"IMPORTANT RULES:\n"
#             f"1. The segment may not explicitly mention a person by name, but may refer to them via context.\n"
#             f"2. If a segment continues discussing the same person from previous segments, assign that person.\n"
#             f"3. If the segment discusses someone's tweet, quote, action, or opinion, assign that person.\n"
#             f"4. If the segment contains a direct quote, assign the person who said it.\n"
#             f"5. Assign 'Narrator' ONLY if the segment truly doesn't discuss any specific person.\n"
#             f"6. DO NOT default to 'Narrator' - carefully analyze context to identify who is being discussed.\n"
#             f"7. Return ONLY the format 'SEGMENT X: PersonName' for each segment."
#             f"8. NEVER assign 'Clip' unless the segment is literally a [CLIP:...] marker."
#         )
#         # Ask GPT
#         response = await openai_client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": combined_text}
#             ],
#         )

#         response_lines = response.choices[0].message.content.strip().split("\n")
#         results = []
#         segment_char_map = {}

#         for line in response_lines:
#             if not line.strip() or "SEGMENT" not in line:
#                 continue
#             try:
#                 segment_part, character = line.split(":", 1)
#                 segment_num = int(re.search(r'SEGMENT\s+(\d+)', segment_part).group(1))
#                 character = character.strip()

#                 segment_text = batch_contexts[segment_num - 1]["segment"]
#                 is_clip_segment = re.match(r"^\[CLIP:.*?\]$", segment_text)

#                 # ❌ Prevent illegal Clip assignment
#                 if character == "Clip" and not is_clip_segment:
#                     logging.warning(f"⚠️ GPT tried to assign 'Clip' to non-clip segment: {segment_text}")
#                     character = None

#                 # ✅ Enforce character is from known list (or fallback)
#                 if not character or (character != "Clip" and character not in celebrities):
#                     match = difflib.get_close_matches(character or "", celebrities, n=1, cutoff=0.6)
#                     chosen = match[0] if match else celebrities[0]
#                 else:
#                     chosen = character

#                 segment_char_map[segment_num - 1] = chosen

#             except Exception as e:
#                 logging.error(f"Error parsing GPT response line '{line}': {e}")

#         # Ensure all batch segments are returned
#         most_recent = celebrities[0]
#         for i, item in enumerate(batch_contexts):
#             assigned_char = segment_char_map.get(i)

#             if not assigned_char:
#                 # Fallback to previous/next/first character
#                 prev = segment_char_map.get(i - 1)
#                 nxt = segment_char_map.get(i + 1)
#                 assigned_char = prev or nxt or most_recent
#             else:
#                 most_recent = assigned_char

#             results.append({
#                 "segment_index": item["segment_index"],
#                 "segment": item["segment"],
#                 "character": assigned_char
#             })

#         return results

#     except Exception as e:
#         logging.error(f"Error in analyze_batch_with_context fallback logic: {traceback.format_exc()}")
#         return [{
#             "segment_index": item["segment_index"],
#             "segment": item["segment"],
#             "character": celebrities[0]  # Last-resort fallback
#         } for item in batch_contexts]

# def post_process_for_consistency(assigned_segments):
#     """Apply additional rules to ensure consistency in character assignments."""
#     if len(assigned_segments) <= 1:
#         return assigned_segments
    
#     processed = assigned_segments.copy()

#     # Rule 0: Reset continuity after [CLIP:...] segments
#     for i in range(1, len(processed)):
#         if processed[i-1]["character"] == "Clip" and processed[i]["character"] == "Narrator":
#             # Prevent carry-over of character from pre-clip to post-clip
#             continue

#     # Rule 1: Maintain character continuity for quotes and responses
#     for i in range(1, len(processed)):
#         # If this segment contains a quote and previous was a named character
#         if ('"' in processed[i]["text"] or "'" in processed[i]["text"]) and processed[i]["character"] == "Narrator":
#             if processed[i-1]["character"] != "Narrator":
#                 # Check if it's a continuation of the previous speaker
#                 processed[i]["character"] = processed[i-1]["character"]
    
#     # Rule 2: Maintain character context across segments about the same topic
#     i = 0
#     while i < len(processed):
#         if processed[i]["character"] != "Narrator":
#             current_character = processed[i]["character"]
            
#             # Look ahead for Narrator segments that might continue discussing this character
#             j = i + 1
#             context_ended = False
            
#             while j < len(processed) and not context_ended:
#                 # If we've hit another named character, the context has changed
#                 if processed[j]["character"] != "Narrator" and processed[j]["character"] != current_character:
#                     context_ended = True
                
#                 # If we're still in the same context but labeled as Narrator, correct it
#                 elif processed[j]["character"] == "Narrator":
#                     # Check if this might continue discussing the same character
#                     # Look for pronouns, references to actions/statements, etc.
#                     pronouns = ["he", "she", "they", "his", "her", "their", "them"]
#                     context_words = ["tweet", "statement", "said", "mentioned", "post", "comment", "response"]
                    
#                     segment_lower = processed[j]["text"].lower()
                    
#                     # If segment has pronouns or context words, likely continuing about the same person
#                     if any(pronoun in segment_lower.split() for pronoun in pronouns) or \
#                        any(word in segment_lower for word in context_words):
#                         processed[j]["character"] = current_character
                
#                 j += 1
        
#         i += 1
    
#     return processed


# async def analyze_celebrity_content(narration_text, niche):
#     """
#     Analyzes text to identify mentioned celebrities and segment the text with context awareness.
#     If niche is 'original', tags all segments as 'Narrator'.
#     Returns segments with identified celebrities or 'Narrator'.
#     """
#     if not narration_text or not isinstance(narration_text, str):
#         raise ValueError("Invalid narration text provided")

#     try:
#         # 🔁 If niche is 'original', return entire narration broken into segments tagged as 'Narrator'
#         if niche == "original":
#             segments = estimate_segments_atomic(narration_text, niche)
#             return [{"character": "Narrator", "text": seg} for seg in segments]

#         # 1. Extract celebrity names
#         all_celebrities = await extract_character_names(narration_text, niche)
#         logging.info(f"Extracted celebrities: {all_celebrities}")

#         if not all_celebrities:
#             return [{"character": "Narrator", "text": narration_text}]

#         # 2. Estimate segments
#         segments = estimate_segments(narration_text, niche)
#         if not segments:
#             return [{"character": "Narrator", "text": narration_text}]

#         # 3. Context-aware celebrity assignment
#         return await process_segments_with_context(segments, all_celebrities)

#     except Exception as e:
#         logging.error(f"Error analyzing celebrity content: {traceback.format_exc()}")
#         return [{"character": "Narrator", "text": narration_text}]



# def get_main_characters(segments, min_mentions=1):
#     """Return main characters with their frequency of mentions."""
#     character_counts = Counter(
#         segment["character"] for segment in segments if segment["character"] not in ("Narrator", "Clip")
#     )

#     # Filter out characters mentioned fewer times than min_mentions
#     main_characters = {
#         char: count for char, count in character_counts.items() if count >= min_mentions
#     }

#     return main_characters





# async def segment_narration(narration_text, niche, characters=None):
#     """
#     Main function to segment narration and assign characters with context awareness.
#     If characters are not provided, it will extract them from the text.
#     """
#     try:
#         if not characters:
#             # Step 1: Analyze narration initially to get character-tagged segments
#             initial_segments = await analyze_celebrity_content(narration_text, niche)

#             # Step 2: Extract main characters with mention counts
#             character_counts = get_main_characters(initial_segments)
#             character_names = list(character_counts.keys())
#             logging.info(f"Main characters identified: {character_counts}")

#             # Step 3: If any characters found, re-analyze with context-aware assignment
#             if character_names:
#                 segments = estimate_segments(narration_text,niche)
#                 return await process_segments_with_context(segments, character_names)
#             else:
#                 return initial_segments
#         else:
#             # Use provided characters
#             segments = estimate_segments(narration_text,niche)
#             return await process_segments_with_context(segments, characters)

#     except Exception as e:
#         logging.error(f"Error in segment_narration: {traceback.format_exc()}")
#         return [{"character": "Narrator", "text": narration_text}]


# async def generate_youtube_keywords_for_characters(text: str, characters: list, max_keywords: int = 5) -> dict:
#     """
#     Generates YouTube search keywords for each character in the list based on the input text.
    
#     Args:
#         text (str): Input text to analyze
#         characters (list): List of characters/entities to generate keywords for
#         api_key (str): Your OpenAI API key
#         max_keywords (int): Maximum number of keywords to return per character (default: 5)
    
#     Returns:
#         dict: A dictionary where keys are characters and values are lists of keywords
#     """
    
#     # Dictionary to store keywords for each character
#     character_keywords = {}
    
#     for character in characters:
#         prompt = f"""Analyze the following text and generate {max_keywords} relevant YouTube search keywords
#         specifically related to {character}. Focus on keywords that would be effective for finding related video content.
#         Format the response as a comma-separated list without numbers or bullet points.
        
#         Text: {text[:500]}  # Truncate to 3000 chars to avoid token limit
        
#         Keywords:"""
        
#         try:
#             response = await openai_client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": "You are a helpful SEO assistant."},
#                     {"role": "user", "content": prompt}
#                 ],
#             )
            
#             if response.choices and response.choices[0].message.content:
#                 raw_output = response.choices[0].message.content.strip()
#                 # Clean and split keywords into a list
               
#                 keywords = [k.strip() for k in re.split(r',|\n', raw_output) if k.strip()]
#                 character_keywords[character] = keywords[:max_keywords]
#             else:
#                 character_keywords[character] = []  # Empty list if no keywords are generated
                
#         except Exception as e:
#             print(f"Error generating keywords for {character}: {e}")
#             character_keywords[character] = []  # Empty list in case of errors
    
#     return character_keywords


# async def extract_main_character_keywords_and_visuals_async(intro, body=None, niche="original"):
#     """Processes intro+body as ONE intro section."""
#     # 0) Combine text (body optional)
#     combined = (intro or "").strip()
#     if body and body.strip():
#         combined = f"{combined}\n\n{body.strip()}"

#     with tqdm(total=3, desc="Extracting Intro (single section)", unit="task") as pbar:
#         # 1) Segment once
#         intro_segments = await segment_narration(combined, niche)
#         pbar.update(1)

#         # 2) Main characters once
#         intro_characters_dict = get_main_characters(intro_segments)
#         pbar.update(1)

#         # 3) YT keywords once
#         yt_intro_keywords = await generate_youtube_keywords_for_characters(
#             combined, list(intro_characters_dict.keys())
#         )
#         pbar.update(1)

#     # Return ONLY the intro block; no body key at all
#     extracted_data = {
#         "intro": {
#             "main_characters": intro_characters_dict,
#             "yt_keywords": yt_intro_keywords,
#             "segments": intro_segments
#         }
#     }

#     print("\nExtraction Completed ✅ (single-section intro)")
#     return extracted_data

# def extract_main_character_keywords_and_visuals(intro, body=None, niche="original"):
#     return asyncio.run(extract_main_character_keywords_and_visuals_async(intro, body, niche))










import os
import asyncio
import openai
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import logging
import re
import difflib
import traceback
from collections import Counter

# Load API credentials
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
def clean_text(text):
    """Clean the input text by removing unnecessary spaces and newlines."""
    if not text or not isinstance(text, str):
        return ""

    # Remove extra spaces and newlines
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text




import re

def estimate_segments(narration_text,niche,min_words=25, absorb_threshold=10):
    if not narration_text or not isinstance(narration_text, str):
        return []

    narration_text = narration_text.strip()

    # Step 1: Split out [CLIP:...] blocks
    clip_pattern = re.compile(r"(\[\s*CLIP\s*:.*?\])", flags=re.IGNORECASE | re.DOTALL)
    parts = clip_pattern.split(narration_text)

    raw_segments = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if re.match(clip_pattern, part):
            raw_segments.append(part)
        else:
            # Protect known patterns
            protected = (
                part.replace("...", " ELLIPSIS ")
                     .replace("Mr.", "MrPROTECTED.")
                     .replace("Mrs.", "MrsPROTECTED.")
                     .replace("Dr.", "DrPROTECTED.")
                     .replace("e.g.", "ePROTECTEDg.")
                     .replace("i.e.", "iPROTECTEDe.")
            )

            words = protected.split()
            buffer = []
            word_count = 0

            for i, word in enumerate(words):
                buffer.append(word)
                word_count += 1

                end_of_sentence = re.search(r'[.!?]$', word)
                strong_end = end_of_sentence and not word.endswith(("MrPROTECTED.", "DrPROTECTED.", "ePROTECTEDg.", "iPROTECTEDe."))
                next_word = words[i+1] if i+1 < len(words) else ""
                next_starts_quote = next_word.startswith('"')

                if word_count >= min_words and (strong_end or next_starts_quote):
                    segment = " ".join(buffer)
                    segment = segment.replace("ELLIPSIS", "...")\
                                     .replace("MrPROTECTED.", "Mr.")\
                                     .replace("MrsPROTECTED.", "Mrs.")\
                                     .replace("DrPROTECTED.", "Dr.")\
                                     .replace("ePROTECTEDg.", "e.g.")\
                                     .replace("iPROTECTEDe.", "i.e.")
                    raw_segments.append(segment.strip())
                    buffer = []
                    word_count = 0

            if buffer:
                segment = " ".join(buffer)
                segment = segment.replace("ELLIPSIS", "...")\
                                 .replace("MrPROTECTED.", "Mr.")\
                                 .replace("MrsPROTECTED.", "Mrs.")\
                                 .replace("DrPROTECTED.", "Dr.")\
                                 .replace("ePROTECTEDg.", "e.g.")\
                                 .replace("iPROTECTEDe.", "i.e.")
                if not segment.endswith(('.', '!', '?', ':')):
                    segment += '.'
                raw_segments.append(segment.strip())

    # Step 2: Post-process to absorb short segments
    final_segments = []
    i = 0

    while i < len(raw_segments):
        current = raw_segments[i]
        is_clip = bool(re.match(clip_pattern, current))
        word_count = len(current.split())

        if word_count >= absorb_threshold or is_clip:
            final_segments.append(current)
            i += 1
            continue

        # It's a short narration segment
        prev_idx = len(final_segments) - 1
        next_idx = i + 1 if i + 1 < len(raw_segments) else None

        prev_is_clip = re.match(clip_pattern, final_segments[prev_idx]) if prev_idx >= 0 else False
        next_is_clip = re.match(clip_pattern, raw_segments[next_idx]) if next_idx is not None else False

        if prev_idx >= 0 and not prev_is_clip:
            # Merge into previous segment
            final_segments[prev_idx] += ' ' + current
        elif next_idx is not None and not next_is_clip:
            # Prepend to next segment
            raw_segments[next_idx] = current + ' ' + raw_segments[next_idx]
        else:
            # Keep as-is (either at start or surrounded by clips)
            final_segments.append(current)

        i += 1

    return final_segments

import re
from typing import List

import re
from typing import List


def estimate_segments_atomic(
    narration_text: str,
    niche: str | None = None,
    min_words: int = 0,        # kept for API compatibility; not used
    absorb_threshold: int = 0  # kept for API compatibility; not used
) -> List[str]:
    """
    Split strictly at major punctuation: . ? ! ; : — –
    Preserves [CLIP: ...] blocks intact and protects common abbreviations.
    """

    if not narration_text or not isinstance(narration_text, str):
        return []

    text = narration_text.strip()
    if not text:
        return []

    # --- Preserve CLIP blocks exactly ---
    clip_re = re.compile(r"(\[\s*CLIP\s*:.*?\])", re.IGNORECASE | re.DOTALL)
    parts = clip_re.split(text)

    # --- Protect abbreviations to avoid false splits on '.' ---
    def protect(s: str) -> str:
        return (s.replace("...", " ELLIPSIS ")
                 .replace("Mr.", "MrPROTECTED.")
                 .replace("Mrs.", "MrsPROTECTED.")
                 .replace("Ms.", "MsPROTECTED.")
                 .replace("Dr.", "DrPROTECTED.")
                 .replace("Prof.", "ProfPROTECTED.")
                 .replace("e.g.", "ePROTECTEDg.")
                 .replace("i.e.", "iPROTECTEDe.")
                 .replace("U.S.", "USPROTECTED.")
                 .replace("etc.", "etcPROTECTED."))

    def unprotect(s: str) -> str:
        return (s.replace("ELLIPSIS", "...")
                 .replace("MrPROTECTED.", "Mr.")
                 .replace("MrsPROTECTED.", "Mrs.")
                 .replace("MsPROTECTED.", "Ms.")
                 .replace("DrPROTECTED.", "Dr.")
                 .replace("ProfPROTECTED.", "Prof.")
                 .replace("ePROTECTEDg.", "e.g.")
                 .replace("iPROTECTEDe.", "i.e.")
                 .replace("USPROTECTED.", "U.S.")
                 .replace("etcPROTECTED.", "etc."))

    # --- Major punctuation split pattern ---
    # Split *after* any of these: . ? ! ; : — – (keep punctuation with the segment)
    SPLIT_PATTERN = re.compile(r"""
        (.*?                # non-greedy up to the first terminator
         [\.!?;:\u2014\u2013]   # . ? ! ; : — (em dash) – (en dash)
        )
        (?:\s+|$)           # followed by whitespace or end
    """, re.VERBOSE | re.DOTALL)

    def split_major_punct(s: str) -> List[str]:
        s = protect(s.strip())
        segments = []
        pos = 0
        for m in SPLIT_PATTERN.finditer(s):
            seg = m.group(1).strip()
            if seg:
                segments.append(seg)
            pos = m.end()
        tail = s[pos:].strip()
        if tail:
            segments.append(tail)
        # Unprotect + tidy + ensure terminal punctuation for non-empty segments
        cleaned = []
        for seg in segments:
            seg = unprotect(seg).strip()
            if not seg:
                continue
            # ensure terminal punctuation (if missing)
            if seg[-1] not in ".!?;:\u2014\u2013":
                seg += "."
            cleaned.append(seg)
        return cleaned

    # --- Process parts and assemble final list ---
    final_segments: List[str] = []
    for part in parts:
        if not part or not part.strip():
            continue
        if clip_re.fullmatch(part.strip()):
            final_segments.append(part.strip())  # keep clip as-is
        else:
            final_segments.extend(split_major_punct(part))

    return final_segments



import re
from typing import List

def estimate_long_segments(
    narration_text: str,
    max_words_per_segment: int = 50,
    min_words_per_segment: int = 20
) -> List[str]:
    """
    Splits narration text into balanced chunks for image generation.
    """

    # --- Ensure numeric types ---
    try:
        max_words_per_segment = int(max_words_per_segment)
        min_words_per_segment = int(min_words_per_segment)
    except ValueError:
        max_words_per_segment, min_words_per_segment = 50, 20

    if not narration_text or not isinstance(narration_text, str):
        return []

    text = narration_text.strip()
    if not text:
        return []

    clip_re = re.compile(r"(\[\s*CLIP\s*:.*?\])", re.IGNORECASE | re.DOTALL)

    def protect(s: str) -> str:
        return (s.replace("...", " ELLIPSIS ")
                 .replace("Mr.", "MrPROTECTED.")
                 .replace("Mrs.", "MrsPROTECTED.")
                 .replace("Ms.", "MsPROTECTED.")
                 .replace("Dr.", "DrPROTECTED.")
                 .replace("Prof.", "ProfPROTECTED.")
                 .replace("e.g.", "ePROTECTEDg.")
                 .replace("i.e.", "iPROTECTEDe.")
                 .replace("U.S.", "USPROTECTED.")
                 .replace("etc.", "etcPROTECTED."))

    def unprotect(s: str) -> str:
        return (s.replace("ELLIPSIS", "...")
                 .replace("MrPROTECTED.", "Mr.")
                 .replace("MrsPROTECTED.", "Mrs.")
                 .replace("MsPROTECTED.", "Ms.")
                 .replace("DrPROTECTED.", "Dr.")
                 .replace("ProfPROTECTED.", "Prof.")
                 .replace("ePROTECTEDg.", "e.g.")
                 .replace("iPROTECTEDe.", "i.e.")
                 .replace("USPROTECTED.", "U.S.")
                 .replace("etcPROTECTED.", "etc."))

    SPLIT_PATTERN = re.compile(r"(?<=[\.!?;:\u2014\u2013])\s+")

    def chunk_sentences(s: str) -> List[str]:
        s = protect(s)
        sentences = [seg.strip() for seg in SPLIT_PATTERN.split(s) if seg.strip()]
        sentences = [unprotect(seg) for seg in sentences]
        return sentences

    def make_balanced_segments(sentences: List[str]) -> List[str]:
        segments = []
        buffer, count = [], 0
        for sent in sentences:
            words = sent.split()
            # 🔹 ensure numeric comparison works
            if count + len(words) > max_words_per_segment and count >= min_words_per_segment:
                segments.append(" ".join(buffer).strip())
                buffer, count = [], 0
            buffer.append(sent)
            count += len(words)
        if buffer:
            segments.append(" ".join(buffer).strip())
        return segments

    final_segments: List[str] = []
    for part in clip_re.split(text):
        if not part.strip():
            continue
        if clip_re.fullmatch(part.strip()):
            final_segments.append(part.strip())
        else:
            sents = chunk_sentences(part)
            final_segments.extend(make_balanced_segments(sents))

    return final_segments



async def extract_character_names(text,niche):
    """Extract potential celebrity names from text using NER via the API."""
    try:
        if (niche=='history'):
            system_prompt = (
             "-You are an expert in analyzing historical texts and identifying key entities.\n"
             "-Extract ONLY the names of real historical figures, countries, empires, governments, or organizations mentioned in the text.\n"
             "-Return each entity on a new line.\n"
             "-Include ONLY the names—no extra commentary, numbers, or punctuation.\n"
             "-For people, return full names if possible. If only a partial name is provided, use context to infer the full name.\n"
             "-Do NOT include titles (e.g., 'King', 'President'), nicknames, or non-historical references.\n"
             "-Do NOT include '[CLIP:...]' markers or any mention of video clips as characters.\n"
             "-Focus only on historically relevant names that played significant roles in events.\n"
             "-Exclude vague or general terms like 'the army', 'the people', or 'the leader'."
        )
        if(niche=='philosophy'):
             print("In philosophy character")
             system_prompt = (
            "-You are a script analysis assistant for a video generation AI.\n"
            "-From the given text, extract the most relevant terms that should be visually represented.\n"
            "-Include:\n"
            "  • Historical or philosophical figures (e.g., 'Socrates', 'Tesla')\n"
            "  • Abstract ideas or themes (e.g., 'ego', 'truth', 'mirror effect')\n"
            "  • Important action verbs that reflect the tone or movement (e.g., 'reject', 'reflect', 'challenge')\n"
            "-Do NOT include punctuation,'[CLIP:...]' markers, or filler words.\n"
            "-Keep everything in a clean, ready-to-use keyword list.\n"
        )
        else:
           system_prompt = (
    "-You are an expert at identifying named entities (people and organizations) in text.\n"
    "-Extract ONLY the names of:\n"
    "  • People: full names of real celebrities, politicians, public figures, or fictional characters.\n"
    "  • Organizations: media outlets (e.g., CNN), companies, governments, political parties, NGOs, teams, institutions, platforms.\n"
    "-Return each name on a new line.\n"
    "-Include ONLY the names, with no additional text, numbers, or punctuation.\n"
    "-For people: Provide full names only. If only a single name is provided, use context to guess the full name; if uncertain, omit.\n"
    "-For organizations: Provide canonical names or widely used acronyms (e.g., CNN, NATO).\n"
    "-Do not include titles, nicknames, handles, hashtags, or partial names.\n"
    "-Do not include generic groups or concepts (e.g., the media, government, fans) unless they are recognized proper organizations.\n"
    "-Normalize near-duplicates (e.g., US, U.S. → United States when referring to the government).\n"
    "-List each unique entity once, in order of first appearance.\n"
    "-Do Not include Clip as Character name.\n"
)


        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
        )

        extracted_names = response.choices[0].message.content.strip().split('\n')
        # Clean up and validate names
        validated_names = []
        for name in extracted_names:
            name = name.strip()
            # Only accept names with alphabetic characters
            if name and re.match(r'^[A-Za-z\s\'-]+$', name):
                validated_names.append(name)
        
        return validated_names
    
    except Exception as e:
        logging.error(f"Error extracting celebrity names: {e}")
        return []

async def process_segments_with_context(segments, celebrities):
    """Process segments while maintaining context across multiple segments.
    
    - Assigns [CLIP:...] segments the default character "Clip"
    - Processes other segments while maintaining context
    """
    if not segments:
        return []

    try:
        # Prepare context windows (up to 5 segments at a time)
        context_windows = []
        for i in range(len(segments)):
            start = max(0, i-2)
            end = min(len(segments), i+3)
            context = " ".join(segments[start:end])

            # Identify if the segment is a [CLIP:...] marker
            if re.match(r"^\[CLIP:.*?\]$", segments[i]):
                character = "Clip"
                segment_text = segments[i]  # Ensure segment text is stored for Clip
            else:
                character = "Narrator" if not celebrities else None  # To be assigned later
                segment_text = segments[i]

            context_windows.append({
                "segment_index": i,
                "context": context,
                "segment": segment_text,  # Ensure segment text is present
                "character": character  # Pre-set for clips
            })

        # Process only the non-[CLIP:...] segments in batches
        batch_size = 10
        all_results = []

        non_clip_batches = [
            item for item in context_windows if item["character"] is None
        ]

        for i in range(0, len(non_clip_batches), batch_size):
            batch = non_clip_batches[i:min(i+batch_size, len(non_clip_batches))]
            batch_results = await analyze_batch_with_context(batch, celebrities)
            all_results.extend(batch_results)

        # Merge processed results with pre-set clip segments
        for item in context_windows:
            if item["character"] == "Clip":
                all_results.append({
                    "segment_index": item["segment_index"],
                    "character": "Clip",
                    "text": item["segment"]  # ✅ Fix: Ensure Clip segments have "text"
                })

        # Sort results by original segment index
        all_results.sort(key=lambda x: x["segment_index"])

        # ✅ Fix: Guarantee every item has "text"
        return post_process_for_consistency([{
            "character": item["character"],
            "text": item.get("text", item.get("segment", "[MISSING TEXT]"))  # Prevents KeyError
        } for item in all_results])

    except Exception as e:
        logging.error(f"Error processing segments with context: {traceback.format_exc()}")
        return [{"character": "Clip" if re.match(r'^\[CLIP:.*?\]$', segment) else "Narrator", "text": segment} for segment in segments]



async def analyze_batch_with_context(batch_contexts, celebrities):
    """Assigns characters using GPT with strict Clip enforcement and smart fallback."""
    try:
        # Format input for GPT
        context_texts = []
        for i, item in enumerate(batch_contexts):
            context_texts.append(
                f"SEGMENT {i+1} CONTEXT:\n{item['context']}\nSEGMENT TO ANALYZE:\n{item['segment']}"
            )

        combined_text = "\n\n".join(context_texts)
        celebrity_list = ", ".join(celebrities)

   
        
        system_prompt = (
            f"You are an expert in analyzing text and determining who is being discussed.\n\n"
            f"For each numbered segment, determine which person from this list is being discussed: {celebrity_list}.\n\n"
            f"IMPORTANT RULES:\n"
            f"1. The segment may not explicitly mention a person by name, but may refer to them via context.\n"
            f"2. If a segment continues discussing the same person from previous segments, assign that person.\n"
            f"3. If the segment discusses someone's tweet, quote, action, or opinion, assign that person.\n"
            f"4. If the segment contains a direct quote, assign the person who said it.\n"
            f"5. Assign 'Narrator' ONLY if the segment truly doesn't discuss any specific person.\n"
            f"6. DO NOT default to 'Narrator' - carefully analyze context to identify who is being discussed.\n"
            f"7. Return ONLY the format 'SEGMENT X: PersonName' for each segment."
            f"8. NEVER assign 'Clip' unless the segment is literally a [CLIP:...] marker."
        )
        # Ask GPT
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_text}
            ],
        )

        response_lines = response.choices[0].message.content.strip().split("\n")
        results = []
        segment_char_map = {}

        for line in response_lines:
            if not line.strip() or "SEGMENT" not in line:
                continue
            try:
                segment_part, character = line.split(":", 1)
                segment_num = int(re.search(r'SEGMENT\s+(\d+)', segment_part).group(1))
                character = character.strip()

                segment_text = batch_contexts[segment_num - 1]["segment"]
                is_clip_segment = re.match(r"^\[CLIP:.*?\]$", segment_text)

                # ❌ Prevent illegal Clip assignment
                if character == "Clip" and not is_clip_segment:
                    logging.warning(f"⚠️ GPT tried to assign 'Clip' to non-clip segment: {segment_text}")
                    character = None

                # ✅ Enforce character is from known list (or fallback)
                if not character or (character != "Clip" and character not in celebrities):
                    match = difflib.get_close_matches(character or "", celebrities, n=1, cutoff=0.6)
                    chosen = match[0] if match else celebrities[0]
                else:
                    chosen = character

                segment_char_map[segment_num - 1] = chosen

            except Exception as e:
                logging.error(f"Error parsing GPT response line '{line}': {e}")

        # Ensure all batch segments are returned
        most_recent = celebrities[0]
        for i, item in enumerate(batch_contexts):
            assigned_char = segment_char_map.get(i)

            if not assigned_char:
                # Fallback to previous/next/first character
                prev = segment_char_map.get(i - 1)
                nxt = segment_char_map.get(i + 1)
                assigned_char = prev or nxt or most_recent
            else:
                most_recent = assigned_char

            results.append({
                "segment_index": item["segment_index"],
                "segment": item["segment"],
                "character": assigned_char
            })

        return results

    except Exception as e:
        logging.error(f"Error in analyze_batch_with_context fallback logic: {traceback.format_exc()}")
        return [{
            "segment_index": item["segment_index"],
            "segment": item["segment"],
            "character": celebrities[0]  # Last-resort fallback
        } for item in batch_contexts]

def post_process_for_consistency(assigned_segments):
    """Apply additional rules to ensure consistency in character assignments."""
    if len(assigned_segments) <= 1:
        return assigned_segments
    
    processed = assigned_segments.copy()

    # Rule 0: Reset continuity after [CLIP:...] segments
    for i in range(1, len(processed)):
        if processed[i-1]["character"] == "Clip" and processed[i]["character"] == "Narrator":
            # Prevent carry-over of character from pre-clip to post-clip
            continue

    # Rule 1: Maintain character continuity for quotes and responses
    for i in range(1, len(processed)):
        # If this segment contains a quote and previous was a named character
        if ('"' in processed[i]["text"] or "'" in processed[i]["text"]) and processed[i]["character"] == "Narrator":
            if processed[i-1]["character"] != "Narrator":
                # Check if it's a continuation of the previous speaker
                processed[i]["character"] = processed[i-1]["character"]
    
    # Rule 2: Maintain character context across segments about the same topic
    i = 0
    while i < len(processed):
        if processed[i]["character"] != "Narrator":
            current_character = processed[i]["character"]
            
            # Look ahead for Narrator segments that might continue discussing this character
            j = i + 1
            context_ended = False
            
            while j < len(processed) and not context_ended:
                # If we've hit another named character, the context has changed
                if processed[j]["character"] != "Narrator" and processed[j]["character"] != current_character:
                    context_ended = True
                
                # If we're still in the same context but labeled as Narrator, correct it
                elif processed[j]["character"] == "Narrator":
                    # Check if this might continue discussing the same character
                    # Look for pronouns, references to actions/statements, etc.
                    pronouns = ["he", "she", "they", "his", "her", "their", "them"]
                    context_words = ["tweet", "statement", "said", "mentioned", "post", "comment", "response"]
                    
                    segment_lower = processed[j]["text"].lower()
                    
                    # If segment has pronouns or context words, likely continuing about the same person
                    if any(pronoun in segment_lower.split() for pronoun in pronouns) or \
                       any(word in segment_lower for word in context_words):
                        processed[j]["character"] = current_character
                
                j += 1
        
        i += 1
    
    return processed


async def analyze_celebrity_content(narration_text, niche,split_mode):
    """
    Analyzes text to identify mentioned celebrities and segment the text with context awareness.
    If niche is 'original', tags all segments as 'Narrator'.
    Returns segments with identified celebrities or 'Narrator'.
    """
    if not narration_text or not isinstance(narration_text, str):
        raise ValueError("Invalid narration text provided")

    try:
        # 🔁 If niche is 'original', return entire narration broken into segments tagged as 'Narrator'
        if niche == "original":
            if split_mode=="atomic":
               segments = estimate_segments_atomic(narration_text, niche)

            if split_mode=="natural":
               segments=estimate_segments(narration_text,niche)

            if split_mode=="extended":
               segments=estimate_long_segments(narration_text,niche)


            return [{"character": "Narrator", "text": seg} for seg in segments]

        # 1. Extract celebrity names
        all_celebrities = await extract_character_names(narration_text, niche)
        logging.info(f"Extracted celebrities: {all_celebrities}")

        if not all_celebrities:
            return [{"character": "Narrator", "text": narration_text}]

        # 2. Estimate segments
        segments = estimate_segments(narration_text, niche)
        if not segments:
            return [{"character": "Narrator", "text": narration_text}]

        # 3. Context-aware celebrity assignment
        return await process_segments_with_context(segments, all_celebrities)

    except Exception as e:
        logging.error(f"Error analyzing celebrity content: {traceback.format_exc()}")
        return [{"character": "Narrator", "text": narration_text}]



def get_main_characters(segments, min_mentions=1):
    """Return main characters with their frequency of mentions."""
    character_counts = Counter(
        segment["character"] for segment in segments if segment["character"] not in ("Narrator", "Clip")
    )

    # Filter out characters mentioned fewer times than min_mentions
    main_characters = {
        char: count for char, count in character_counts.items() if count >= min_mentions
    }

    return main_characters





async def segment_narration(narration_text, niche,split_mode, characters=None):
    """
    Main function to segment narration and assign characters with context awareness.
    If characters are not provided, it will extract them from the text.
    """
    try:
        if not characters:
            # Step 1: Analyze narration initially to get character-tagged segments
            initial_segments = await analyze_celebrity_content(narration_text, niche,split_mode)

            # Step 2: Extract main characters with mention counts
            character_counts = get_main_characters(initial_segments)
            character_names = list(character_counts.keys())
            logging.info(f"Main characters identified: {character_counts}")

            # Step 3: If any characters found, re-analyze with context-aware assignment
            if character_names:
                segments = estimate_segments(narration_text,niche)
                return await process_segments_with_context(segments, character_names)
            else:
                return initial_segments
        else:
            # Use provided characters
            segments = estimate_segments(narration_text,niche)
            return await process_segments_with_context(segments, characters)

    except Exception as e:
        logging.error(f"Error in segment_narration: {traceback.format_exc()}")
        return [{"character": "Narrator", "text": narration_text}]


async def generate_youtube_keywords_for_characters(text: str, characters: list, max_keywords: int = 5) -> dict:
    """
    Generates YouTube search keywords for each character in the list based on the input text.
    
    Args:
        text (str): Input text to analyze
        characters (list): List of characters/entities to generate keywords for
        api_key (str): Your OpenAI API key
        max_keywords (int): Maximum number of keywords to return per character (default: 5)
    
    Returns:
        dict: A dictionary where keys are characters and values are lists of keywords
    """
    
    # Dictionary to store keywords for each character
    character_keywords = {}
    
    for character in characters:
        prompt = f"""Analyze the following text and generate {max_keywords} relevant YouTube search keywords
        specifically related to {character}. Focus on keywords that would be effective for finding related video content.
        Format the response as a comma-separated list without numbers or bullet points.
        
        Text: {text[:500]}  # Truncate to 3000 chars to avoid token limit
        
        Keywords:"""
        
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful SEO assistant."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            if response.choices and response.choices[0].message.content:
                raw_output = response.choices[0].message.content.strip()
                # Clean and split keywords into a list
               
                keywords = [k.strip() for k in re.split(r',|\n', raw_output) if k.strip()]
                character_keywords[character] = keywords[:max_keywords]
            else:
                character_keywords[character] = []  # Empty list if no keywords are generated
                
        except Exception as e:
            print(f"Error generating keywords for {character}: {e}")
            character_keywords[character] = []  # Empty list in case of errors
    
    return character_keywords


async def extract_main_character_keywords_and_visuals_async(intro, body=None,split_mode=None, niche="original"):
    """Processes intro+body as ONE intro section."""
    # 0) Combine text (body optional)
    combined = (intro or "").strip()
    if body and body.strip():
        combined = f"{combined}\n\n{body.strip()}"

    with tqdm(total=3, desc="Extracting Intro (single section)", unit="task") as pbar:
        # 1) Segment once
        intro_segments = await segment_narration(combined, niche,split_mode)
        pbar.update(1)

        # 2) Main characters once
        intro_characters_dict = get_main_characters(intro_segments)
        pbar.update(1)

        # 3) YT keywords once
        yt_intro_keywords = await generate_youtube_keywords_for_characters(
            combined, list(intro_characters_dict.keys())
        )
        pbar.update(1)

    # Return ONLY the intro block; no body key at all
    extracted_data = {
        "intro": {
            "main_characters": intro_characters_dict,
            "yt_keywords": yt_intro_keywords,
            "segments": intro_segments
        }
    }

    print("\nExtraction Completed ✅ (single-section intro)")
    return extracted_data

def extract_main_character_keywords_and_visuals(intro, body=None, split_mode=None,niche="original"):
    print("In extrcat in sections")
    return asyncio.run(extract_main_character_keywords_and_visuals_async(intro, body,split_mode, niche))


# def extract_main_character_keywords_and_visuals(intro, body,niche):
#     """Runs the async function synchronously in a normal Python environment."""
#     return asyncio.run(extract_main_character_keywords_and_visuals_async(intro, body,niche))


# Example Usage
# intro_text = """
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
# Can’t wait to see your thoughts on this!"""




# body_text = "Tesla's latest announcement has left investors speechless. Meanwhile, Jeff Bezos is reportedly keeping a close eye on Musk's next move."

# character_visual_data = extract_main_character_keywords_and_visuals(intro_text, body_text)
# # intro_characters=character_visual_data['intro']['main_characters']
# print(character_visual_data)

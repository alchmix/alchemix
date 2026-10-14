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

# # Load API credentials
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize OpenAI client
# openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
# def clean_text(text):
#     """Clean the input text by removing unnecessary spaces and newlines."""
#     if not text or not isinstance(text, str):
#         return ""

#     # Remove extra spaces and newlines
#     cleaned_text = re.sub(r'\s+', ' ', text).strip()
#     return cleaned_text




# import re



# def _word_boundary_cut(s, target):
#     if len(s) <= target:
#         return len(s)
#     idx = min(target, len(s))
#     while idx > 0 and not s[idx - 1].isspace():
#         idx -= 1
#     if idx == 0:
#         idx = min(target, len(s))
#         while idx < len(s) and not s[idx].isspace():
#             idx += 1
#     return idx


# def _split_narration_block_by_chars(text, max_chars=1900, min_tail=1000):
#     """Split one narration block (no clips inside). Whole block <= max_chars stays as ONE segment."""
#     text = text.strip()
#     if not text:
#         return []
#     if len(text) <= max_chars:
#         return [text]

#     def rebalance_evenly(block):
#         block = block.strip()
#         if len(block) <= max_chars:
#             return [block]
#         mid = len(block) // 2
#         cut = _word_boundary_cut(block, mid)
#         if cut == 0 or cut >= len(block):
#             cut = _word_boundary_cut(block, max_chars)
#         part1 = block[:cut].strip()
#         part2 = block[cut:].strip()
#         segments = []
#         for part in (part1, part2):
#             if not part:
#                 continue
#             if len(part) <= max_chars:
#                 segments.append(part)
#             else:
#                 segments.extend(split_block(part))
#         return segments

#     def split_block(block):
#         block = block.strip()
#         if len(block) <= max_chars:
#             return [block]
#         cut = _word_boundary_cut(block, max_chars)
#         first = block[:cut].strip()
#         rest = block[cut:].strip()
#         if not rest:
#             return [first] if first else []
#         if len(rest) >= min_tail:
#             segments = [first] if first else []
#             segments.extend(split_block(rest))
#             return segments
#         return rebalance_evenly(block)

#     return split_block(text)


# def estimate_segments_tts_char_cap(narration_text, max_chars=1900, min_tail=1000):
#     """
#     TTS segmentation for Commentary + Original.
#     - Clips stay atomic
#     - Narration split only between clips, then by char cap
#     - Block <= max_chars → always 1 segment
#     """
#     if not narration_text or not isinstance(narration_text, str):
#         return []

#     text = narration_text.strip()
#     if not text:
#         return []

#     clip_re = re.compile(r"(\[\s*CLIP\s*:.*?\])", re.IGNORECASE | re.DOTALL)
#     parts = clip_re.split(text)
#     final_segments = []

#     for part in parts:
#         part = part.strip()
#         if not part:
#             continue
#         if clip_re.fullmatch(part):
#             final_segments.append(part)
#         else:
#             final_segments.extend(_split_narration_block_by_chars(part, max_chars, min_tail))

#     return final_segments

# def estimate_segments(narration_text,niche,min_words=25, absorb_threshold=10):
#     if not narration_text or not isinstance(narration_text, str):
#         return []

#     narration_text = narration_text.strip()

#     # Step 1: Split out [CLIP:...] blocks
#     clip_pattern = re.compile(r"(\[\s*CLIP\s*:.*?\])", flags=re.IGNORECASE | re.DOTALL)
#     parts = clip_pattern.split(narration_text)

#     raw_segments = []

#     for part in parts:
#         part = part.strip()
#         if not part:
#             continue

#         if re.match(clip_pattern, part):
#             raw_segments.append(part)
#         else:
#             # Protect known patterns
#             protected = (
#                 part.replace("...", " ELLIPSIS ")
#                      .replace("Mr.", "MrPROTECTED.")
#                      .replace("Mrs.", "MrsPROTECTED.")
#                      .replace("Dr.", "DrPROTECTED.")
#                      .replace("e.g.", "ePROTECTEDg.")
#                      .replace("i.e.", "iPROTECTEDe.")
#             )

#             words = protected.split()
#             buffer = []
#             word_count = 0

#             for i, word in enumerate(words):
#                 buffer.append(word)
#                 word_count += 1

#                 end_of_sentence = re.search(r'[.!?]$', word)
#                 strong_end = end_of_sentence and not word.endswith(("MrPROTECTED.", "DrPROTECTED.", "ePROTECTEDg.", "iPROTECTEDe."))
#                 next_word = words[i+1] if i+1 < len(words) else ""
#                 next_starts_quote = next_word.startswith('"')

#                 if word_count >= min_words and (strong_end or next_starts_quote):
#                     segment = " ".join(buffer)
#                     segment = segment.replace("ELLIPSIS", "...")\
#                                      .replace("MrPROTECTED.", "Mr.")\
#                                      .replace("MrsPROTECTED.", "Mrs.")\
#                                      .replace("DrPROTECTED.", "Dr.")\
#                                      .replace("ePROTECTEDg.", "e.g.")\
#                                      .replace("iPROTECTEDe.", "i.e.")
#                     raw_segments.append(segment.strip())
#                     buffer = []
#                     word_count = 0

#             if buffer:
#                 segment = " ".join(buffer)
#                 segment = segment.replace("ELLIPSIS", "...")\
#                                  .replace("MrPROTECTED.", "Mr.")\
#                                  .replace("MrsPROTECTED.", "Mrs.")\
#                                  .replace("DrPROTECTED.", "Dr.")\
#                                  .replace("ePROTECTEDg.", "e.g.")\
#                                  .replace("iPROTECTEDe.", "i.e.")
#                 if not segment.endswith(('.', '!', '?', ':')):
#                     segment += '.'
#                 raw_segments.append(segment.strip())

#     # Step 2: Post-process to absorb short segments
#     final_segments = []
#     i = 0

#     while i < len(raw_segments):
#         current = raw_segments[i]
#         is_clip = bool(re.match(clip_pattern, current))
#         word_count = len(current.split())

#         if word_count >= absorb_threshold or is_clip:
#             final_segments.append(current)
#             i += 1
#             continue

#         # It's a short narration segment
#         prev_idx = len(final_segments) - 1
#         next_idx = i + 1 if i + 1 < len(raw_segments) else None

#         prev_is_clip = re.match(clip_pattern, final_segments[prev_idx]) if prev_idx >= 0 else False
#         next_is_clip = re.match(clip_pattern, raw_segments[next_idx]) if next_idx is not None else False

#         if prev_idx >= 0 and not prev_is_clip:
#             # Merge into previous segment
#             final_segments[prev_idx] += ' ' + current
#         elif next_idx is not None and not next_is_clip:
#             # Prepend to next segment
#             raw_segments[next_idx] = current + ' ' + raw_segments[next_idx]
#         else:
#             # Keep as-is (either at start or surrounded by clips)
#             final_segments.append(current)

#         i += 1

#     return final_segments

# import re
# from typing import List

# import re
# from typing import List


# def estimate_segments_atomic(
#     narration_text: str,
#     niche: str | None = None,
#     min_words: int = 0,        # kept for API compatibility; not used
#     absorb_threshold: int = 0  # kept for API compatibility; not used
# ) -> List[str]:
#     """
#     Split strictly at major punctuation: . ? ! ; : — –
#     Preserves [CLIP: ...] blocks intact and protects common abbreviations.
#     """

#     if not narration_text or not isinstance(narration_text, str):
#         return []

#     text = narration_text.strip()
#     if not text:
#         return []

#     # --- Preserve CLIP blocks exactly ---
#     clip_re = re.compile(r"(\[\s*CLIP\s*:.*?\])", re.IGNORECASE | re.DOTALL)
#     parts = clip_re.split(text)

#     # --- Protect abbreviations to avoid false splits on '.' ---
#     def protect(s: str) -> str:
#         return (s.replace("...", " ELLIPSIS ")
#                  .replace("Mr.", "MrPROTECTED.")
#                  .replace("Mrs.", "MrsPROTECTED.")
#                  .replace("Ms.", "MsPROTECTED.")
#                  .replace("Dr.", "DrPROTECTED.")
#                  .replace("Prof.", "ProfPROTECTED.")
#                  .replace("e.g.", "ePROTECTEDg.")
#                  .replace("i.e.", "iPROTECTEDe.")
#                  .replace("U.S.", "USPROTECTED.")
#                  .replace("etc.", "etcPROTECTED."))

#     def unprotect(s: str) -> str:
#         return (s.replace("ELLIPSIS", "...")
#                  .replace("MrPROTECTED.", "Mr.")
#                  .replace("MrsPROTECTED.", "Mrs.")
#                  .replace("MsPROTECTED.", "Ms.")
#                  .replace("DrPROTECTED.", "Dr.")
#                  .replace("ProfPROTECTED.", "Prof.")
#                  .replace("ePROTECTEDg.", "e.g.")
#                  .replace("iPROTECTEDe.", "i.e.")
#                  .replace("USPROTECTED.", "U.S.")
#                  .replace("etcPROTECTED.", "etc."))

#     # --- Major punctuation split pattern ---
#     # Split *after* any of these: . ? ! ; : — – (keep punctuation with the segment)
#     SPLIT_PATTERN = re.compile(r"""
#         (.*?                # non-greedy up to the first terminator
#          [\.!?;:\u2014\u2013]   # . ? ! ; : — (em dash) – (en dash)
#         )
#         (?:\s+|$)           # followed by whitespace or end
#     """, re.VERBOSE | re.DOTALL)

#     def split_major_punct(s: str) -> List[str]:
#         s = protect(s.strip())
#         segments = []
#         pos = 0
#         for m in SPLIT_PATTERN.finditer(s):
#             seg = m.group(1).strip()
#             if seg:
#                 segments.append(seg)
#             pos = m.end()
#         tail = s[pos:].strip()
#         if tail:
#             segments.append(tail)
#         # Unprotect + tidy + ensure terminal punctuation for non-empty segments
#         cleaned = []
#         for seg in segments:
#             seg = unprotect(seg).strip()
#             if not seg:
#                 continue
#             # ensure terminal punctuation (if missing)
#             if seg[-1] not in ".!?;:\u2014\u2013":
#                 seg += "."
#             cleaned.append(seg)
#         return cleaned

#     # --- Process parts and assemble final list ---
#     final_segments: List[str] = []
#     for part in parts:
#         if not part or not part.strip():
#             continue
#         if clip_re.fullmatch(part.strip()):
#             final_segments.append(part.strip())  # keep clip as-is
#         else:
#             final_segments.extend(split_major_punct(part))

#     return final_segments



# import re
# from typing import List

# def estimate_long_segments(
#     narration_text: str,
#     max_words_per_segment: int = 50,
#     min_words_per_segment: int = 20
# ) -> List[str]:
#     """
#     Splits narration text into balanced chunks for image generation.
#     """

#     # --- Ensure numeric types ---
#     try:
#         max_words_per_segment = int(max_words_per_segment)
#         min_words_per_segment = int(min_words_per_segment)
#     except ValueError:
#         max_words_per_segment, min_words_per_segment = 50, 20

#     if not narration_text or not isinstance(narration_text, str):
#         return []

#     text = narration_text.strip()
#     if not text:
#         return []

#     clip_re = re.compile(r"(\[\s*CLIP\s*:.*?\])", re.IGNORECASE | re.DOTALL)

#     def protect(s: str) -> str:
#         return (s.replace("...", " ELLIPSIS ")
#                  .replace("Mr.", "MrPROTECTED.")
#                  .replace("Mrs.", "MrsPROTECTED.")
#                  .replace("Ms.", "MsPROTECTED.")
#                  .replace("Dr.", "DrPROTECTED.")
#                  .replace("Prof.", "ProfPROTECTED.")
#                  .replace("e.g.", "ePROTECTEDg.")
#                  .replace("i.e.", "iPROTECTEDe.")
#                  .replace("U.S.", "USPROTECTED.")
#                  .replace("etc.", "etcPROTECTED."))

#     def unprotect(s: str) -> str:
#         return (s.replace("ELLIPSIS", "...")
#                  .replace("MrPROTECTED.", "Mr.")
#                  .replace("MrsPROTECTED.", "Mrs.")
#                  .replace("MsPROTECTED.", "Ms.")
#                  .replace("DrPROTECTED.", "Dr.")
#                  .replace("ProfPROTECTED.", "Prof.")
#                  .replace("ePROTECTEDg.", "e.g.")
#                  .replace("iPROTECTEDe.", "i.e.")
#                  .replace("USPROTECTED.", "U.S.")
#                  .replace("etcPROTECTED.", "etc."))

#     SPLIT_PATTERN = re.compile(r"(?<=[\.!?;:\u2014\u2013])\s+")

#     def chunk_sentences(s: str) -> List[str]:
#         s = protect(s)
#         sentences = [seg.strip() for seg in SPLIT_PATTERN.split(s) if seg.strip()]
#         sentences = [unprotect(seg) for seg in sentences]
#         return sentences

#     def make_balanced_segments(sentences: List[str]) -> List[str]:
#         segments = []
#         buffer, count = [], 0
#         for sent in sentences:
#             words = sent.split()
#             # 🔹 ensure numeric comparison works
#             if count + len(words) > max_words_per_segment and count >= min_words_per_segment:
#                 segments.append(" ".join(buffer).strip())
#                 buffer, count = [], 0
#             buffer.append(sent)
#             count += len(words)
#         if buffer:
#             segments.append(" ".join(buffer).strip())
#         return segments

#     final_segments: List[str] = []
#     for part in clip_re.split(text):
#         if not part.strip():
#             continue
#         if clip_re.fullmatch(part.strip()):
#             final_segments.append(part.strip())
#         else:
#             sents = chunk_sentences(part)
#             final_segments.extend(make_balanced_segments(sents))

#     return final_segments



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


# async def analyze_celebrity_content(narration_text, niche, split_mode, commentary_mode=None):
#     """
#     Analyzes text to identify mentioned celebrities and segment the text with context awareness.
#     If niche is 'original', tags all segments as 'Narrator'.
#     If niche is 'commentary' + commentary_mode 'original', uses TTS char-cap segments, no character splitting.
#     """
#     if not narration_text or not isinstance(narration_text, str):
#         raise ValueError("Invalid narration text provided")

#     try:
#         # Commentary + Original → char-cap TTS segments, no multi-character logic
#         if niche == "commentary" and commentary_mode == "original":
#             segments = estimate_segments_tts_char_cap(narration_text, max_chars=1900, min_tail=1000)
#             return [{"character": "Narrator", "text": seg} for seg in segments]

#         # Original niche → split_mode based segments, all Narrator
#         if niche == "original":
#             if split_mode == "atomic":
#                 segments = estimate_segments_atomic(narration_text, niche)
#             elif split_mode == "natural":
#                 segments = estimate_segments(narration_text, niche)
#             elif split_mode == "extended":
#                 segments = estimate_long_segments(narration_text, niche)
#             else:
#                 segments = estimate_segments_atomic(narration_text, niche)
#             return [{"character": "Narrator", "text": seg} for seg in segments]

#         # All other niches — existing character + word-based segmentation
#         all_celebrities = await extract_character_names(narration_text, niche)
#         logging.info(f"Extracted celebrities: {all_celebrities}")

#         if not all_celebrities:
#             return [{"character": "Narrator", "text": narration_text}]

#         segments = estimate_segments(narration_text, niche)
#         if not segments:
#             return [{"character": "Narrator", "text": narration_text}]

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



# async def segment_narration(narration_text, niche, split_mode, characters=None, commentary_mode=None):
#     """
#     Main function to segment narration and assign characters with context awareness.
#     Commentary + Original skips character re-segmentation entirely.
#     """
#     try:
#         if niche == "commentary" and commentary_mode == "original":
#             return await analyze_celebrity_content(narration_text, niche, split_mode, commentary_mode)

#         if niche == "original":
#             return await analyze_celebrity_content(narration_text, niche, split_mode, commentary_mode)

#         if not characters:
#             initial_segments = await analyze_celebrity_content(narration_text, niche, split_mode, commentary_mode)
#             character_counts = get_main_characters(initial_segments)
#             character_names = list(character_counts.keys())
#             logging.info(f"Main characters identified: {character_counts}")

#             if character_names:
#                 segments = estimate_segments(narration_text, niche)
#                 return await process_segments_with_context(segments, character_names)
#             return initial_segments

#         segments = estimate_segments(narration_text, niche)
#         return await process_segments_with_context(segments, characters)

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


# async def extract_main_character_keywords_and_visuals_async(
#     intro, body=None, split_mode=None, niche="original", commentary_mode=None
# ):
#     combined = (intro or "").strip()
#     if body and body.strip():
#         combined = f"{combined}\n\n{body.strip()}"

#     is_commentary_original = niche == "commentary" and commentary_mode == "original"

#     with tqdm(total=3, desc="Extracting Intro (single section)", unit="task") as pbar:
#         intro_segments = await segment_narration(
#             combined, niche, split_mode, commentary_mode=commentary_mode
#         )
#         pbar.update(1)

#         intro_characters_dict = get_main_characters(intro_segments)

#         # Auto-create Narrator for commentary + original
#         if is_commentary_original:
#             narrator_count = sum(
#                 1 for s in intro_segments if s.get("character") == "Narrator"
#             )
#             intro_characters_dict = {"Narrator": max(narrator_count, 1)}

#         pbar.update(1)

#         if is_commentary_original or not intro_characters_dict:
#             yt_intro_keywords = {}
#         else:
#             yt_intro_keywords = await generate_youtube_keywords_for_characters(
#                 combined, list(intro_characters_dict.keys())
#             )
#         pbar.update(1)

#     extracted_data = {
#         "intro": {
#             "main_characters": intro_characters_dict,
#             "yt_keywords": yt_intro_keywords,
#             "segments": intro_segments,
#         }
#     }

#     print("\nExtraction Completed ✅ (single-section intro)")
#     return extracted_data

# def extract_main_character_keywords_and_visuals(
#     intro, body=None, split_mode=None, niche="original", commentary_mode=None
# ):
#     print("In extrcat in sections")
#     return asyncio.run(
#         extract_main_character_keywords_and_visuals_async(
#             intro, body, split_mode, niche, commentary_mode
#         )
#     )


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
from typing import List

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

# Speechify /v1/audio/speech — commentary + original only
COMMENTARY_ORIGINAL_MAX_CHARS = 2000
COMMENTARY_ORIGINAL_MIN_TAIL = 1000

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_text(text):
    """Clean the input text by removing unnecessary spaces and newlines."""
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Commentary + original: char-cap splitting at paragraph / sentence boundaries
# ---------------------------------------------------------------------------

_ABBREV_PROTECT = (
    ("...", " ELLIPSIS "),
    ("Mr.", "MrPROTECTED."),
    ("Mrs.", "MrsPROTECTED."),
    ("Ms.", "MsPROTECTED."),
    ("Dr.", "DrPROTECTED."),
    ("Prof.", "ProfPROTECTED."),
    ("e.g.", "ePROTECTEDg."),
    ("i.e.", "iPROTECTEDe."),
    ("U.S.", "USPROTECTED."),
    ("etc.", "etcPROTECTED."),
)

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:\u2014\u2013])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;\u2014\u2013])\s+")


def _protect_abbreviations(s: str) -> str:
    for old, new in _ABBREV_PROTECT:
        s = s.replace(old, new)
    return s


def _unprotect_abbreviations(s: str) -> str:
    for old, new in _ABBREV_PROTECT:
        s = s.replace(new, old)
    return s


def _split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT.split(text.strip()) if p.strip()]


def _split_into_sentences(paragraph: str) -> list[str]:
    protected = _protect_abbreviations(paragraph.strip())
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(protected) if p.strip()]
    return [_unprotect_abbreviations(p) for p in parts]


def _join_parts(parts: list[str]) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def _split_oversized_sentence(sentence: str, max_chars: int) -> list[str]:
    """
    Last resort when one sentence still exceeds max_chars.
    Try clause boundaries first, then word boundaries.
    """
    sentence = sentence.strip()
    if len(sentence) <= max_chars:
        return [sentence]

    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]
    if len(clauses) > 1:
        out: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for clause in clauses:
            add = len(clause) + (1 if buf else 0)
            if buf and buf_len + add > max_chars:
                out.append(_join_parts(buf))
                buf, buf_len = [clause], len(clause)
            else:
                buf.append(clause)
                buf_len += add
        if buf:
            out.append(_join_parts(buf))
        if all(len(x) <= max_chars for x in out):
            return out

    out, buf, buf_len = [], [], 0
    for word in sentence.split():
        add = len(word) + (1 if buf else 0)
        if buf and buf_len + add > max_chars:
            out.append(" ".join(buf))
            buf, buf_len = [word], len(word)
        else:
            buf.append(word)
            buf_len += add
    if buf:
        out.append(" ".join(buf))
    return out


def _pack_units(units: list[str], max_chars: int, min_tail: int) -> list[str]:
    """Pack sentence/clause units without crossing boundaries."""
    if not units:
        return []

    segments: list[str] = []
    i = 0

    while i < len(units):
        chunk: list[str] = []
        chunk_len = 0
        j = i

        while j < len(units):
            unit = units[j]
            add_len = len(unit) + (1 if chunk else 0)

            if len(unit) > max_chars:
                break
            if chunk and chunk_len + add_len > max_chars:
                break

            chunk.append(unit)
            chunk_len += add_len
            j += 1

        if not chunk:
            segments.extend(_split_oversized_sentence(units[i], max_chars))
            i += 1
            continue

        rest_units = units[j:]
        rest_len = sum(len(u) + 1 for u in rest_units)

        if rest_units and rest_len < min_tail and len(chunk) > 1:
            moved = chunk.pop()
            j -= 1
            rest_units = [moved] + rest_units

        segments.append(_join_parts(chunk))
        i = j

    return segments


def _split_narration_block_by_chars(
    text,
    max_chars=COMMENTARY_ORIGINAL_MAX_CHARS,
    min_tail=COMMENTARY_ORIGINAL_MIN_TAIL,
):
    """
    Split one narration block (no clips inside).
    Cuts at paragraph → sentence boundaries; never mid-sentence when avoidable.
    Used only for commentary + original (Speechify char cap).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    final: list[str] = []
    paragraphs = _split_into_paragraphs(text) or [text]

    for para in paragraphs:
        if len(para) <= max_chars:
            final.append(para)
            continue

        sentences = _split_into_sentences(para) or [para]
        final.extend(_pack_units(sentences, max_chars, min_tail))

    return final


def estimate_segments_tts_char_cap(
    narration_text,
    max_chars=COMMENTARY_ORIGINAL_MAX_CHARS,
    min_tail=COMMENTARY_ORIGINAL_MIN_TAIL,
):
    """
    TTS segmentation for Commentary + Original only.
    - Clips stay atomic
    - Narration split only between clips, then by char cap at sentence boundaries
    - Block <= max_chars → always 1 segment
    """
    if not narration_text or not isinstance(narration_text, str):
        return []

    text = narration_text.strip()
    if not text:
        return []

    clip_re = re.compile(r"(\[\s*CLIP\s*:.*?\])", re.IGNORECASE | re.DOTALL)
    parts = clip_re.split(text)
    final_segments = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if clip_re.fullmatch(part):
            final_segments.append(part)
        else:
            final_segments.extend(
                _split_narration_block_by_chars(part, max_chars, min_tail)
            )

    return final_segments


# ---------------------------------------------------------------------------
# Other split modes (no Speechify char cap)
# ---------------------------------------------------------------------------

def estimate_segments(narration_text, niche, min_words=25, absorb_threshold=10):
    if not narration_text or not isinstance(narration_text, str):
        return []

    narration_text = narration_text.strip()
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

                end_of_sentence = re.search(r"[.!?]$", word)
                strong_end = end_of_sentence and not word.endswith(
                    ("MrPROTECTED.", "DrPROTECTED.", "ePROTECTEDg.", "iPROTECTEDe.")
                )
                next_word = words[i + 1] if i + 1 < len(words) else ""
                next_starts_quote = next_word.startswith('"')

                if word_count >= min_words and (strong_end or next_starts_quote):
                    segment = " ".join(buffer)
                    segment = (
                        segment.replace("ELLIPSIS", "...")
                        .replace("MrPROTECTED.", "Mr.")
                        .replace("MrsPROTECTED.", "Mrs.")
                        .replace("DrPROTECTED.", "Dr.")
                        .replace("ePROTECTEDg.", "e.g.")
                        .replace("iPROTECTEDe.", "i.e.")
                    )
                    raw_segments.append(segment.strip())
                    buffer = []
                    word_count = 0

            if buffer:
                segment = " ".join(buffer)
                segment = (
                    segment.replace("ELLIPSIS", "...")
                    .replace("MrPROTECTED.", "Mr.")
                    .replace("MrsPROTECTED.", "Mrs.")
                    .replace("DrPROTECTED.", "Dr.")
                    .replace("ePROTECTEDg.", "e.g.")
                    .replace("iPROTECTEDe.", "i.e.")
                )
                if not segment.endswith((".", "!", "?", ":")):
                    segment += "."
                raw_segments.append(segment.strip())

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

        prev_idx = len(final_segments) - 1
        next_idx = i + 1 if i + 1 < len(raw_segments) else None

        prev_is_clip = (
            re.match(clip_pattern, final_segments[prev_idx]) if prev_idx >= 0 else False
        )
        next_is_clip = (
            re.match(clip_pattern, raw_segments[next_idx]) if next_idx is not None else False
        )

        if prev_idx >= 0 and not prev_is_clip:
            final_segments[prev_idx] += " " + current
        elif next_idx is not None and not next_is_clip:
            raw_segments[next_idx] = current + " " + raw_segments[next_idx]
        else:
            final_segments.append(current)

        i += 1

    return final_segments


def estimate_segments_atomic(
    narration_text: str,
    niche: str | None = None,
    min_words: int = 0,
    absorb_threshold: int = 0,
) -> List[str]:
    """Split at major punctuation. Preserves [CLIP: ...] blocks intact."""
    if not narration_text or not isinstance(narration_text, str):
        return []

    text = narration_text.strip()
    if not text:
        return []

    clip_re = re.compile(r"(\[\s*CLIP\s*:.*?\])", re.IGNORECASE | re.DOTALL)
    parts = clip_re.split(text)

    def protect(s: str) -> str:
        return (
            s.replace("...", " ELLIPSIS ")
            .replace("Mr.", "MrPROTECTED.")
            .replace("Mrs.", "MrsPROTECTED.")
            .replace("Ms.", "MsPROTECTED.")
            .replace("Dr.", "DrPROTECTED.")
            .replace("Prof.", "ProfPROTECTED.")
            .replace("e.g.", "ePROTECTEDg.")
            .replace("i.e.", "iPROTECTEDe.")
            .replace("U.S.", "USPROTECTED.")
            .replace("etc.", "etcPROTECTED.")
        )

    def unprotect(s: str) -> str:
        return (
            s.replace("ELLIPSIS", "...")
            .replace("MrPROTECTED.", "Mr.")
            .replace("MrsPROTECTED.", "Mrs.")
            .replace("MsPROTECTED.", "Ms.")
            .replace("DrPROTECTED.", "Dr.")
            .replace("ProfPROTECTED.", "Prof.")
            .replace("ePROTECTEDg.", "e.g.")
            .replace("iPROTECTEDe.", "i.e.")
            .replace("USPROTECTED.", "U.S.")
            .replace("etcPROTECTED.", "etc.")
        )

    split_pattern = re.compile(
        r"""
        (.*?
         [\.!?;:\u2014\u2013]
        )
        (?:\s+|$)
        """,
        re.VERBOSE | re.DOTALL,
    )

    def split_major_punct(s: str) -> List[str]:
        s = protect(s.strip())
        segments = []
        pos = 0
        for m in split_pattern.finditer(s):
            seg = m.group(1).strip()
            if seg:
                segments.append(seg)
            pos = m.end()
        tail = s[pos:].strip()
        if tail:
            segments.append(tail)

        cleaned = []
        for seg in segments:
            seg = unprotect(seg).strip()
            if not seg:
                continue
            if seg[-1] not in ".!?;:\u2014\u2013":
                seg += "."
            cleaned.append(seg)
        return cleaned

    final_segments: List[str] = []
    for part in parts:
        if not part or not part.strip():
            continue
        if clip_re.fullmatch(part.strip()):
            final_segments.append(part.strip())
        else:
            final_segments.extend(split_major_punct(part))

    return final_segments


def estimate_long_segments(
    narration_text: str,
    max_words_per_segment: int = 50,
    min_words_per_segment: int = 20,
) -> List[str]:
    """Splits narration text into balanced chunks for image generation."""
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
        return (
            s.replace("...", " ELLIPSIS ")
            .replace("Mr.", "MrPROTECTED.")
            .replace("Mrs.", "MrsPROTECTED.")
            .replace("Ms.", "MsPROTECTED.")
            .replace("Dr.", "DrPROTECTED.")
            .replace("Prof.", "ProfPROTECTED.")
            .replace("e.g.", "ePROTECTEDg.")
            .replace("i.e.", "iPROTECTEDe.")
            .replace("U.S.", "USPROTECTED.")
            .replace("etc.", "etcPROTECTED.")
        )

    def unprotect(s: str) -> str:
        return (
            s.replace("ELLIPSIS", "...")
            .replace("MrPROTECTED.", "Mr.")
            .replace("MrsPROTECTED.", "Mrs.")
            .replace("MsPROTECTED.", "Ms.")
            .replace("DrPROTECTED.", "Dr.")
            .replace("ProfPROTECTED.", "Prof.")
            .replace("ePROTECTEDg.", "e.g.")
            .replace("iPROTECTEDe.", "i.e.")
            .replace("USPROTECTED.", "U.S.")
            .replace("etcPROTECTED.", "etc.")
        )

    split_pattern = re.compile(r"(?<=[\.!?;:\u2014\u2013])\s+")

    def chunk_sentences(s: str) -> List[str]:
        s = protect(s)
        sentences = [seg.strip() for seg in split_pattern.split(s) if seg.strip()]
        return [unprotect(seg) for seg in sentences]

    def make_balanced_segments(sentences: List[str]) -> List[str]:
        segments = []
        buffer, count = [], 0
        for sent in sentences:
            words = sent.split()
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


# ---------------------------------------------------------------------------
# Character extraction + assignment
# ---------------------------------------------------------------------------

async def extract_character_names(text, niche):
    """Extract potential celebrity names from text using NER via the API."""
    try:
        if niche == "history":
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
        elif niche == "philosophy":
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
                {"role": "user", "content": text},
            ],
        )

        extracted_names = response.choices[0].message.content.strip().split("\n")
        validated_names = []
        for name in extracted_names:
            name = name.strip()
            if name and re.match(r"^[A-Za-z\s\'-]+$", name):
                validated_names.append(name)

        return validated_names

    except Exception as e:
        logging.error(f"Error extracting celebrity names: {e}")
        return []


async def process_segments_with_context(segments, celebrities):
    """Assign characters while preserving [CLIP:...] segments as Clip."""
    if not segments:
        return []

    try:
        context_windows = []
        for i in range(len(segments)):
            start = max(0, i - 2)
            end = min(len(segments), i + 3)
            context = " ".join(segments[start:end])

            if re.match(r"^\[CLIP:.*?\]$", segments[i]):
                character = "Clip"
                segment_text = segments[i]
            else:
                character = "Narrator" if not celebrities else None
                segment_text = segments[i]

            context_windows.append({
                "segment_index": i,
                "context": context,
                "segment": segment_text,
                "character": character,
            })

        batch_size = 10
        all_results = []
        non_clip_batches = [item for item in context_windows if item["character"] is None]

        for i in range(0, len(non_clip_batches), batch_size):
            batch = non_clip_batches[i : min(i + batch_size, len(non_clip_batches))]
            batch_results = await analyze_batch_with_context(batch, celebrities)
            all_results.extend(batch_results)

        for item in context_windows:
            if item["character"] == "Clip":
                all_results.append({
                    "segment_index": item["segment_index"],
                    "character": "Clip",
                    "text": item["segment"],
                })

        all_results.sort(key=lambda x: x["segment_index"])

        return post_process_for_consistency([
            {
                "character": item["character"],
                "text": item.get("text", item.get("segment", "[MISSING TEXT]")),
            }
            for item in all_results
        ])

    except Exception:
        logging.error(f"Error processing segments with context: {traceback.format_exc()}")
        return [
            {
                "character": "Clip" if re.match(r"^\[CLIP:.*?\]$", segment) else "Narrator",
                "text": segment,
            }
            for segment in segments
        ]


async def analyze_batch_with_context(batch_contexts, celebrities):
    """Assigns characters using GPT with strict Clip enforcement and smart fallback."""
    try:
        context_texts = []
        for i, item in enumerate(batch_contexts):
            context_texts.append(
                f"SEGMENT {i + 1} CONTEXT:\n{item['context']}\nSEGMENT TO ANALYZE:\n{item['segment']}"
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

        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_text},
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
                segment_num = int(re.search(r"SEGMENT\s+(\d+)", segment_part).group(1))
                character = character.strip()

                segment_text = batch_contexts[segment_num - 1]["segment"]
                is_clip_segment = re.match(r"^\[CLIP:.*?\]$", segment_text)

                if character == "Clip" and not is_clip_segment:
                    logging.warning(
                        f"GPT tried to assign 'Clip' to non-clip segment: {segment_text}"
                    )
                    character = None

                if not character or (character != "Clip" and character not in celebrities):
                    match = difflib.get_close_matches(character or "", celebrities, n=1, cutoff=0.6)
                    chosen = match[0] if match else celebrities[0]
                else:
                    chosen = character

                segment_char_map[segment_num - 1] = chosen

            except Exception as e:
                logging.error(f"Error parsing GPT response line '{line}': {e}")

        most_recent = celebrities[0]
        for i, item in enumerate(batch_contexts):
            assigned_char = segment_char_map.get(i)

            if not assigned_char:
                prev = segment_char_map.get(i - 1)
                nxt = segment_char_map.get(i + 1)
                assigned_char = prev or nxt or most_recent
            else:
                most_recent = assigned_char

            results.append({
                "segment_index": item["segment_index"],
                "segment": item["segment"],
                "character": assigned_char,
            })

        return results

    except Exception:
        logging.error(f"Error in analyze_batch_with_context: {traceback.format_exc()}")
        return [
            {
                "segment_index": item["segment_index"],
                "segment": item["segment"],
                "character": celebrities[0],
            }
            for item in batch_contexts
        ]


def post_process_for_consistency(assigned_segments):
    """Apply additional rules to ensure consistency in character assignments."""
    if len(assigned_segments) <= 1:
        return assigned_segments

    processed = assigned_segments.copy()

    for i in range(1, len(processed)):
        if (
            '"' in processed[i]["text"] or "'" in processed[i]["text"]
        ) and processed[i]["character"] == "Narrator":
            if processed[i - 1]["character"] != "Narrator":
                processed[i]["character"] = processed[i - 1]["character"]

    i = 0
    while i < len(processed):
        if processed[i]["character"] != "Narrator":
            current_character = processed[i]["character"]
            j = i + 1
            context_ended = False

            while j < len(processed) and not context_ended:
                if (
                    processed[j]["character"] != "Narrator"
                    and processed[j]["character"] != current_character
                ):
                    context_ended = True
                elif processed[j]["character"] == "Narrator":
                    pronouns = ["he", "she", "they", "his", "her", "their", "them"]
                    context_words = [
                        "tweet", "statement", "said", "mentioned",
                        "post", "comment", "response",
                    ]
                    segment_lower = processed[j]["text"].lower()
                    if any(pronoun in segment_lower.split() for pronoun in pronouns) or any(
                        word in segment_lower for word in context_words
                    ):
                        processed[j]["character"] = current_character
                j += 1
        i += 1

    return processed


async def analyze_celebrity_content(narration_text, niche, split_mode, commentary_mode=None):
    """
    Segment narration and assign characters.
    Commentary + original uses Speechify char-cap splitting only.
    """
    if not narration_text or not isinstance(narration_text, str):
        raise ValueError("Invalid narration text provided")

    try:
        if niche == "commentary" and commentary_mode == "original":
            segments = estimate_segments_tts_char_cap(narration_text)
            return [{"character": "Narrator", "text": seg} for seg in segments]

        if niche == "original":
            if split_mode == "atomic":
                segments = estimate_segments_atomic(narration_text, niche)
            elif split_mode == "natural":
                segments = estimate_segments(narration_text, niche)
            elif split_mode == "extended":
                segments = estimate_long_segments(narration_text, niche)
            else:
                segments = estimate_segments_atomic(narration_text, niche)
            return [{"character": "Narrator", "text": seg} for seg in segments]

        all_celebrities = await extract_character_names(narration_text, niche)
        logging.info(f"Extracted celebrities: {all_celebrities}")

        if not all_celebrities:
            return [{"character": "Narrator", "text": narration_text}]

        segments = estimate_segments(narration_text, niche)
        if not segments:
            return [{"character": "Narrator", "text": narration_text}]

        return await process_segments_with_context(segments, all_celebrities)

    except Exception:
        logging.error(f"Error analyzing celebrity content: {traceback.format_exc()}")
        return [{"character": "Narrator", "text": narration_text}]


def get_main_characters(segments, min_mentions=1):
    """Return main characters with their frequency of mentions."""
    character_counts = Counter(
        segment["character"]
        for segment in segments
        if segment["character"] not in ("Narrator", "Clip")
    )
    return {
        char: count for char, count in character_counts.items() if count >= min_mentions
    }


async def segment_narration(
    narration_text, niche, split_mode, characters=None, commentary_mode=None
):
    """Main entry: segment narration and assign characters."""
    try:
        if niche == "commentary" and commentary_mode == "original":
            return await analyze_celebrity_content(
                narration_text, niche, split_mode, commentary_mode
            )

        if niche == "original":
            return await analyze_celebrity_content(
                narration_text, niche, split_mode, commentary_mode
            )

        if not characters:
            initial_segments = await analyze_celebrity_content(
                narration_text, niche, split_mode, commentary_mode
            )
            character_counts = get_main_characters(initial_segments)
            character_names = list(character_counts.keys())
            logging.info(f"Main characters identified: {character_counts}")

            if character_names:
                segments = estimate_segments(narration_text, niche)
                return await process_segments_with_context(segments, character_names)
            return initial_segments

        segments = estimate_segments(narration_text, niche)
        return await process_segments_with_context(segments, characters)

    except Exception:
        logging.error(f"Error in segment_narration: {traceback.format_exc()}")
        return [{"character": "Narrator", "text": narration_text}]


async def generate_youtube_keywords_for_characters(
    text: str, characters: list, max_keywords: int = 5
) -> dict:
    character_keywords = {}

    for character in characters:
        prompt = f"""Analyze the following text and generate {max_keywords} relevant YouTube search keywords
        specifically related to {character}. Focus on keywords that would be effective for finding related video content.
        Format the response as a comma-separated list without numbers or bullet points.

        Text: {text[:500]}

        Keywords:"""

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful SEO assistant."},
                    {"role": "user", "content": prompt},
                ],
            )

            if response.choices and response.choices[0].message.content:
                raw_output = response.choices[0].message.content.strip()
                keywords = [k.strip() for k in re.split(r",|\n", raw_output) if k.strip()]
                character_keywords[character] = keywords[:max_keywords]
            else:
                character_keywords[character] = []

        except Exception as e:
            print(f"Error generating keywords for {character}: {e}")
            character_keywords[character] = []

    return character_keywords


async def extract_main_character_keywords_and_visuals_async(
    intro, body=None, split_mode=None, niche="original", commentary_mode=None
):
    combined = (intro or "").strip()
    if body and body.strip():
        combined = f"{combined}\n\n{body.strip()}"

    is_commentary_original = niche == "commentary" and commentary_mode == "original"

    with tqdm(total=3, desc="Extracting Intro (single section)", unit="task") as pbar:
        intro_segments = await segment_narration(
            combined, niche, split_mode, commentary_mode=commentary_mode
        )
        pbar.update(1)

        intro_characters_dict = get_main_characters(intro_segments)

        if is_commentary_original:
            narrator_count = sum(
                1 for s in intro_segments if s.get("character") == "Narrator"
            )
            intro_characters_dict = {"Narrator": max(narrator_count, 1)}

        pbar.update(1)

        if is_commentary_original or not intro_characters_dict:
            yt_intro_keywords = {}
        else:
            yt_intro_keywords = await generate_youtube_keywords_for_characters(
                combined, list(intro_characters_dict.keys())
            )
        pbar.update(1)

    extracted_data = {
        "intro": {
            "main_characters": intro_characters_dict,
            "yt_keywords": yt_intro_keywords,
            "segments": intro_segments,
        }
    }

    print("\nExtraction Completed ✅ (single-section intro)")
    return extracted_data


def extract_main_character_keywords_and_visuals(
    intro, body=None, split_mode=None, niche="original", commentary_mode=None
):
    print("In extrcat in sections")
    return asyncio.run(
        extract_main_character_keywords_and_visuals_async(
            intro, body, split_mode, niche, commentary_mode
        )
    )

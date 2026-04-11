import os
import re
import difflib
import asyncio
import logging
import traceback
import tiktoken
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# Load API key
load_dotenv()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
encoding = tiktoken.encoding_for_model("gpt-4o")

def token_count(text):
    return len(encoding.encode(text))

def token_split_narration(text, max_tokens=2000):
    words = text.strip().split()
    chunks, current_chunk = [], []
    for word in words:
        current_chunk.append(word)
        if token_count(" ".join(current_chunk)) >= max_tokens:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
    return chunks

def parse_script_with_metadata(script_segments):
    elements = []
    for seg in script_segments:
        text = seg.get("text", "").strip()
        character = seg.get("character")
        tag = seg.get("tag")
        if text.startswith("[CLIP:") and text.endswith("]"):
            elements.append({"type": "clip", "text": text, "character": character, "tag": tag})
        else:
            for chunk in token_split_narration(text):
                elements.append({"type": "narration", "text": chunk, "character": character, "tag": tag})
    return elements

def get_tail_words(text, min_words=20):
    return ' '.join(text.strip().split()[-min_words:]) if text else ''

async def paraphrase_segment(text, context, niche, pbar):
    if not text.strip(): return text

    if niche == 'history':
        prompt = f"""
PARAPHRASE THIS NARRATIVE:
- Keep factual integrity
- Use documentary tone
CONTEXT:
{context}
SEGMENT_TEXT:
{text}
Only return the paraphrased version.
"""
    elif niche == 'philosophy':
        prompt = f"""
PARAPHRASE THIS PHILOSOPHICAL THOUGHT:
- Preserve logical structure
- Reflective tone

CONTEXT:
{context}
SEGMENT_TEXT:
{text}
Only return the paraphrased version.
"""
    else:
        prompt = f"""
PARAPHRASE TASK:
- Same meaning, new phrasing
- Conversational tone
CONTEXT:
{context}
SEGMENT_TEXT:
{text}
Only return the paraphrased version.
"""

    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    pbar.update(1)
    return response.choices[0].message.content.strip()

async def process_script_sections(video_data, niche='default'):
    output_data = {k: video_data[k] for k in ["title", "asset_folder", "video_path"]}
    output_data["sections"] = {}

    for section_name, section in video_data["sections"].items():
        parsed_elements = parse_script_with_metadata(section.get("segment", []))
        paraphrase_tasks = []
        context_memory = []

        with tqdm(total=len(parsed_elements), desc=f"{section_name} paraphrasing") as pbar:
            for elem in parsed_elements:
                if elem["type"] == "narration":
                    context_tail = get_tail_words(" ".join(context_memory[-3:]), 20)
                    task = paraphrase_segment(elem["text"], context_tail, niche, pbar)
                    paraphrase_tasks.append(task)
                    context_memory.append(elem["text"])
                else:
                    paraphrase_tasks.append(asyncio.sleep(0, result=elem["text"]))

            results = await asyncio.gather(*paraphrase_tasks)

        new_segments = []
        for i, elem in enumerate(parsed_elements):
            new_segments.append({
                "text": results[i],
                "character": elem.get("character"),
                "tag": elem.get("tag")
            })

        output_data["sections"][section_name] = {**section, "segment": new_segments}

    return output_data

def estimate_segments(narration_text, min_words=25, absorb_threshold=10):
    if not narration_text or not isinstance(narration_text, str):
        return []

    narration_text = narration_text.strip()
    clip_pattern = re.compile(r"(\[\s*CLIP\s*:.*?\])", flags=re.IGNORECASE | re.DOTALL)

    # Step 1: Split text around [CLIP:...] blocks
    parts = clip_pattern.split(narration_text)
    raw_segments = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if re.match(clip_pattern, part):
            raw_segments.append(part)
            continue

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
            strong_end = end_of_sentence and not word.endswith((
                "MrPROTECTED.", "DrPROTECTED.", "ePROTECTEDg.", "iPROTECTEDe."
            ))
            next_word = words[i+1] if i+1 < len(words) else ""
            next_starts_quote = next_word.startswith('"')

            if word_count >= min_words and (strong_end or next_starts_quote):
                segment = " ".join(buffer)
                segment = segment.replace("ELLIPSIS", "...") \
                                 .replace("MrPROTECTED.", "Mr.") \
                                 .replace("MrsPROTECTED.", "Mrs.") \
                                 .replace("DrPROTECTED.", "Dr.") \
                                 .replace("ePROTECTEDg.", "e.g.") \
                                 .replace("iPROTECTEDe.", "i.e.")
                raw_segments.append(segment.strip())
                buffer = []
                word_count = 0

        if buffer:
            segment = " ".join(buffer)
            segment = segment.replace("ELLIPSIS", "...") \
                             .replace("MrPROTECTED.", "Mr.") \
                             .replace("MrsPROTECTED.", "Mrs.") \
                             .replace("DrPROTECTED.", "Dr.") \
                             .replace("ePROTECTEDg.", "e.g.") \
                             .replace("iPROTECTEDe.", "i.e.")
            if not segment.endswith(('.', '!', '?', ':')):
                segment += '.'
            raw_segments.append(segment.strip())

    # Step 2: Post-process and absorb short narration when legal
    final_segments = []
    i = 0

    while i < len(raw_segments):
        current = raw_segments[i]
        is_clip = bool(re.match(clip_pattern, current))
        word_count = len(current.split())

        if is_clip or word_count >= absorb_threshold:
            final_segments.append(current)
            i += 1
            continue

        # Try to merge narration if it's between two narrations (no clips)
        prev_idx = len(final_segments) - 1
        next_idx = i + 1 if i + 1 < len(raw_segments) else None

        prev_is_clip = re.match(clip_pattern, final_segments[prev_idx]) if prev_idx >= 0 else False
        next_is_clip = re.match(clip_pattern, raw_segments[next_idx]) if next_idx is not None else False

        if prev_idx >= 0 and not prev_is_clip:
            final_segments[prev_idx] += ' ' + current
        elif next_idx is not None and not next_is_clip:
            raw_segments[next_idx] = current + ' ' + raw_segments[next_idx]
        else:
            # ❗Iron Rule: can't merge across clips, so keep short segment as-is
            final_segments.append(current)

        i += 1

    return final_segments

def segment_paraphrased_sections(data):
    segmented_data = {
        "title": data.get("title"),
        "asset_folder": data.get("asset_folder"),
        "video_path": data.get("video_path"),
        "sections": {}
    }

    for section_name, section in data.get("sections", {}).items():
        new_segments = []

        for item in section.get("segment", []):
            text = item.get("text", "").strip()
            character = item.get("character")
            tag = item.get("tag")

            # ✅ Preserve full metadata for [CLIP:...] segments
            if text.startswith("[CLIP:") and text.endswith("]"):
                new_segments.append({
                    "text": text,
                    "character": character,  # ✅ DO NOT overwrite
                    "tag": tag               # ✅ Preserve original tag
                })
                continue

            # ✂️ Segment narration and assign original metadata to each chunk
            for chunk in estimate_segments(text):
                new_segments.append({
                    "text": chunk,
                    "character": character,
                    "tag": tag
                })

        segmented_data["sections"][section_name] = {
            "segment": new_segments,
            "section_characters": section.get("section_characters", [])
        }

    return segmented_data

def find_closest_character(ai_response, valid_characters):
    """
    Find the closest matching character from the valid list using fuzzy matching
    """
    if not ai_response or not valid_characters:
        return valid_characters[0] if valid_characters else "Narrator"
    
    # First try exact match (case-insensitive)
    for char in valid_characters:
        if ai_response.strip().lower() == char.strip().lower():
            return char
    
    # Then try fuzzy matching
    closest_matches = difflib.get_close_matches(
        ai_response.strip(), 
        valid_characters, 
        n=1, 
        cutoff=0.6
    )
    
    return closest_matches[0] if closest_matches else valid_characters[0]

def validate_and_correct_character(character_name, valid_characters):
    """
    Ensure the character name exactly matches one from the valid list
    """
    if not valid_characters:
        return "Narrator"
    
    # Direct match
    if character_name in valid_characters:
        return character_name
    
    # Case-insensitive match
    for valid_char in valid_characters:
        if character_name.lower() == valid_char.lower():
            return valid_char
    
    # Fuzzy matching as fallback
    return find_closest_character(character_name, valid_characters)

async def analyze_batch_with_context(batch, celebrities):
    """
    Enhanced version with strict character validation
    """
    try:
        context_texts = [f"SEGMENT {i+1} CONTEXT:\n{b['context']}\nSEGMENT TO ANALYZE:\n{b['segment']}" for i, b in enumerate(batch)]
        combined_text = "\n\n".join(context_texts)
        
        # Create a numbered list for the prompt to make it super clear
        character_list = "\n".join([f"{i+1}. {char}" for i, char in enumerate(celebrities)])
        
        prompt = (
            f"You are an expert in analyzing text and determining who is being discussed.\n\n"
            f"VALID CHARACTER LIST (USE EXACTLY AS SPELLED):\n{character_list}\n\n"
            f"For each numbered segment, determine which character is being discussed.\n\n"
            f"CRITICAL RULES:\n"
            f"1. You MUST use the EXACT spelling from the character list above\n"
            f"2. Do NOT modify, add accents, or change any character names\n"
            f"3. If unsure, pick the most relevant character from the list\n"
            f"4. Consider context from surrounding segments\n"
            f"5. Look for pronouns, quotes, or actions that reference a character\n"
            f"6. Return format: 'SEGMENT X: [EXACT_CHARACTER_NAME]'\n"
            f"7. Only use characters from the numbered list above\n\n"
            f"TEXT TO ANALYZE:\n{combined_text}"
        )
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise text analyzer. Follow instructions exactly and use only the provided character names without any modifications."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Lower temperature for more consistent responses
        )
        
        lines = response.choices[0].message.content.strip().split("\n")
        result_map = {}
        
        for line in lines:
            if not line.strip() or ":" not in line: 
                continue
            try:
                seg_part, char_part = line.split(":", 1)
                idx = int(re.search(r'SEGMENT\s+(\d+)', seg_part).group(1)) - 1
                raw_character = char_part.strip()
                
                # Validate and correct the character name
                validated_character = validate_and_correct_character(raw_character, celebrities)
                result_map[idx] = validated_character
                
                # Log if correction was made
                if raw_character != validated_character:
                    logging.warning(f"Character corrected: '{raw_character}' -> '{validated_character}'")
                    
            except (ValueError, AttributeError) as e:
                logging.warning(f"Failed to parse line: {line}. Error: {e}")
                continue

        results = []
        most_recent = celebrities[0] if celebrities else "Narrator"
        
        for i, b in enumerate(batch):
            char = result_map.get(i, most_recent)
            # Double-check validation
            char = validate_and_correct_character(char, celebrities)
            most_recent = char
            results.append({
                "segment_index": b["segment_index"], 
                "character": char, 
                "text": b["segment"]
            })
        return results
        
    except Exception as e:
        logging.error(f"Batch analysis failed: {traceback.format_exc()}")
        # Fallback with validation
        fallback_char = celebrities[0] if celebrities else "Narrator"
        return [{
            "segment_index": b["segment_index"], 
            "character": fallback_char, 
            "text": b["segment"]
        } for b in batch]

async def process_segments_with_context(segments, celebrities):
    """
    Enhanced version with additional validation
    """
    try:
        # Validate that celebrities list is not empty
        if not celebrities:
            logging.warning("No celebrities provided, using default narrator")
            celebrities = ["Narrator"]
        
        context_windows = []
        for i in range(len(segments)):
            text = segments[i]["text"]
            context = " ".join([seg["text"] for seg in segments[max(0, i-2):min(len(segments), i+3)]])
            is_clip = text.startswith("[CLIP:") and text.endswith("]")
            has_tag = segments[i].get("tag") is not None
            # 💥 Iron Rule: if it has a tag (but is NOT a clip), skip character assignment
            character = segments[i].get("character") if is_clip or has_tag else None
            context_windows.append({
                "segment_index": i,
                "context": context,
                "segment": text,
                "character": character
            })

        all_results = []

        # Analyze only narration (non-clips)
        non_clip = [item for item in context_windows if item["character"] is None]
        for i in range(0, len(non_clip), 10):
            batch = non_clip[i:i+10]
            batch_results = await analyze_batch_with_context(batch, celebrities)
            all_results.extend(batch_results)

        # Add CLIPs with preserved characters (also validate them)
        for item in context_windows:
            if item["character"] is not None:
                validated_char = validate_and_correct_character(item["character"], celebrities)
                all_results.append({
                    "segment_index": item["segment_index"],
                    "character": validated_char,
                    "text": item["segment"]
                })

        # Sort to original order
        all_results.sort(key=lambda x: x["segment_index"])
        
        # Final validation pass
        final_results = []
        for item in all_results:
            validated_char = validate_and_correct_character(item["character"], celebrities)
            final_results.append({
                "character": validated_char, 
                "text": item["text"]
            })
        
        return final_results

    except Exception as e:
        logging.error(f"Error in process_segments_with_context: {traceback.format_exc()}")
        fallback_char = celebrities[0] if celebrities else "Narrator"
        return [{"character": fallback_char, "text": seg["text"]} for seg in segments]

async def assign_characters_to_segments(data):
    """
    Enhanced version with validation logging
    """
    final_data = {k: data[k] for k in ["title", "asset_folder", "video_path"]}
    final_data["sections"] = {}
    
    for section_name, section in data["sections"].items():
        segments = section.get("segment", [])
        characters = section.get("section_characters", [])
        
        # Log the characters being used for this section
        logging.info(f"Section '{section_name}' characters: {characters}")
        
        if not characters:
            logging.warning(f"No characters found for section '{section_name}', using default")
            characters = ["Narrator"]
        
        updated = await process_segments_with_context(segments, characters)

        # Preserve original tags
        for i, seg in enumerate(segments):
            if i < len(updated):
                updated[i]["tag"] = seg.get("tag")
        
        # Final validation check
        for seg in updated:
            if seg["character"] not in characters:
                logging.error(f"VALIDATION FAILED: Character '{seg['character']}' not in valid list {characters}")
                seg["character"] = characters[0]  # Force correction
        
        final_data["sections"][section_name] = {**section, "segment": updated}
    
    return final_data

def audit_character_assignments(data):
    """
    Audit function to check for character assignment issues
    """
    issues = []
    
    for section_name, section in data.get("sections", {}).items():
        valid_characters = section.get("section_characters", [])
        segments = section.get("segment", [])
        
        for i, seg in enumerate(segments):
            assigned_char = seg.get("character", "")
            if assigned_char not in valid_characters:
                issues.append({
                    "section": section_name,
                    "segment_index": i,
                    "assigned_character": assigned_char,
                    "valid_characters": valid_characters,
                    "text_preview": seg.get("text", "")[:50] + "..."
                })
    
    return issues

async def generate_viral_title_from_intro(paraphrased_data):
    intro_segments = paraphrased_data.get("sections", {}).get("intro", {}).get("segment", [])
    if not intro_segments:
        return "🔥 VIRAL VIDEO ALERT: CONTROVERSY EXPLODES!"

    narration_segments = [seg for seg in intro_segments if not seg["text"].strip().startswith("[CLIP:")]

    # Join narration-only text
    combined_text = " ".join([seg["text"] for seg in narration_segments])
    truncated_intro = combined_text[:200].strip()

    # Determine most frequent character (excluding clips and unknowns)
    char_counts = {}
    for seg in narration_segments:
        char = seg.get("character", "").strip()
        if char and char.lower() not in ["", "clip", "no character", "unknown"]:
            char_counts[char] = char_counts.get(char, 0) + 1

    main_character = max(char_counts.items(), key=lambda x: x[1])[0] if char_counts else "A CONTROVERSIAL FIGURE"

    prompt = f"""
Write a viral YouTube title based on this intro narration:

"{truncated_intro}"

Focus on the main character: {main_character}

Rules:
- Title must reflect the true speaker and theme
- Must be UNDER 70 characters
- Use ALL CAPS for drama
- Include controversy, celebrity names, or hot issues
- Avoid any mention of [CLIP] content
- Only return the rewritten title

Examples:
- "BILL MAHER PRAISES TRUMP?! MEDIA MELTDOWN FOLLOWS"
- "LIBERALS TURN ON BILL MAHER AFTER SHOCKING COMMENT"
- "MAHER UNLEASHES ON WOKE CULTURE IN HBO RANT"

Only return the title:
"""

    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

async def full_pipeline(video_data, niche='default'):
    paraphrased = await process_script_sections(video_data, niche)
    segmented = segment_paraphrased_sections(paraphrased)
    final_output = await assign_characters_to_segments(segmented)

    # Audit for issues
    issues = audit_character_assignments(final_output)
    if issues:
        logging.error(f"Character assignment issues found: {len(issues)} problems")
        for issue in issues:
            logging.error(f"Section: {issue['section']}, Invalid character: '{issue['assigned_character']}', Valid: {issue['valid_characters']}")

    # Generate viral title from paraphrased intro
    viral_title = await generate_viral_title_from_intro(paraphrased)
    final_output["title"] = viral_title
    print(final_output)
    intro = final_output["sections"].get("intro", {})
    for seg in intro.get("segment", [])[:100]:  # Limit to first 5
        print(f"[{seg['character']}]: {seg['text']}\n")

    print("\n✅ SAMPLE OUTPUT (BODY SECTION):")
    intro = final_output["sections"].get("body", {})
    for seg in intro.get("segment", [])[:100]:  # Limit to first 5
        print(f"[{seg['character']}]: {seg['text']}\n")


    

    return final_output

# video_data = {
#     "title": "Final Video",
#    "sections": {
#     "intro": {
#       "section": "intro",
#       "section_script": "[CLIP: Sports by sex doesn't make sense. Yes, it does Because again, we haven't reinvented homo sapiens since crystal Pepsi came out I've spent three decades on TV mocking Republicans who said climate change was just a theory and now I got to deal with people who Say, you know, what else is just a theory by a but that can't happen overnight Because the people who aren't getting the food or the medicine they're the ones who are gonna suffer You don't see why people find this incredibly cruel You said earlier that we're treating it as if it's a it's an overweight person putting them on a crash diet oh, no, the federal government is essentially somebody who's 800 pounds and You cannot survive like this So yes There are some drastic measures that have to be taken and I will add this to you If you think it's bad now if we get into a real debt crisis It will be significantly worse for the American people and other people that we might help around the globe the best thing to do right now Getting quick. You're gonna have to shut some things down triage the area rebuild the stuff That is good that we all can agree with the other stuff. That's hyper political or makes no sense or fraudulent What about the Department of Education now? I I don't know that much about it, but I've never read good things Rahm Emanuel who I agree with almost everything here was a quote He said a third of eighth graders can't read and now he wants to close the department ] Okay everyone, let's dive in! You know Bill Maher, correct? Lately, he's really shaking things up, and honestly, it's the buzz of the town—just thinking about it could send your temper through the roof. That famed funnyman and star of his own HBO show has made a daring move that truly blindsided us all—he’s actually defending Donald Trump! [CLIP: If I give you a billion dollars and you can't tell me what happened to it, that to me is wasteful. That means you are not responsible. But if you can't tell me where it went, then what am I supposed to think? An $850 billion budget to an organization that can't pass an audit and tell you where that money went, like, I think most people would consider that somewhere in the realm of waste, fraud, or abuse because they would wonder why that money isn't well accounted for. But I'm a human being who lives on the earth and can't figure out how $850 billion to a department means that the rank and file still have to be on food stamps. Like to me, that's fucking corruption. ] And can you believe this? After all the mocking he's done over the years, that's the guy. Maher is really going for it, hitting hard at both political parties and he's not holding back at all. [CLIP: With communists, that human was no longer selfish. In America today, that human is no longer born male or female. And obesity is not something that affects health. You can be healthy at any size. Really, we voted on it. A formerly serious magazine last year published with a straight face an article called, Separating Sports by Sex Doesn't Make Sense. Yes, it does. Because again, we haven't reinvented homo sapien since crystal Pepsi came out. I've spent three decades on TV mocking Republicans who said climate change was just a theory. And now I've got to deal with people who say, you know what else is just a theory? ] In the current heated political scene, his recent actions regarding burning issues such as education and immigration have polarized his audience—some are applauding, while others are absolutely livid! Hold up! If you're into content that pushes boundaries and keeps it authentic, don't forget to hit that subscribe button. Go ahead, I’ll wait... Done? Great to have you! Now, let’s dive back into the discussion. Bill Maher's Views Okay, let's delve into... all the juicy details of Bill Maher’s straightforward opinions, okay? Lately, Maher has used his platform to expose his thoughts on some of today’s most pressing social issues. And let me tell you, he does not mince his words. His forthright analysis has been strikingly forthright on topics like the Department of Education, immigration laws in California, and let’s not forget those sensitive debates about gender in sports. Now, let's discuss Bill's perspective on education. How do I phrase this..? For ages, education has been regarded almost as a holy, untouchable domain, right? But here's Maher, calling out issues, pointing fingers, and declaring, \"Something here just doesn’t make sense.\" Frankly, his criticism isn’t without basis—it’s supported by data that can’t be overlooked. Take this alarming statistic, for instance: a third of our nation’s eighth graders aren’t reading at the appropriate grade level. So, there's Maher, virtually yelling, \"What's actually happening here?\" Honestly, are we just sinking billions into an educational system that’s deteriorating year after year? According to Bill, it seems the Department of Education is more of a financial sinkhole than a proponent of education, sort of like a vortex where taxpayer money vanishes. Immigration Policies Moving on to Maher on immigration—here’s where it heats up even more. You see, lawmakers in California have chosen to offer government aid to undocumented immigrants for home purchases. [CLIP: The California lawmakers just passed a law giving government assistance to undocumented immigrants to buy houses. That's kind of a different place than the Democratic Party used to be on immigration, is it not? Well, let me just say, immigration had always been a bipartisan issue. I refer you to the- But not free houses. Well, that's not free housing. It's the American dream, being available to more people. California's always in the lead. Maybe others will follow that lead, but that's up to those states. So you'd vote for this law? I'm not familiar with exactly what it is, but making the American dream of home ownership available to all people is something we have to do for people who are here now. This is for the undocumented. ]",
#       "imageData": [
#         {
#           "characterName": "Clip",
#           "images": [
#             "https://live.staticflickr.com/65535/54286148307_32f9bc32c0_o.png",
#             "https://live.staticflickr.com/3128/2658597429_ae262f6c1f_o.jpg",
#             "https://live.staticflickr.com/6080/6096823038_84902028a4_o.jpg",
#             "https://live.staticflickr.com/65535/54358769196_216ce103a7_o.png"
#           ]
#         },
#         {
#           "characterName": "Bill Maher",
#           "images": [
#             "https://i.ticketweb.com/i/00/09/56/65/63/Original.jpg?v=1",
#             "https://dw-wp-production.imgix.net/2025/03/GettyImages-2202961976-scaled.jpg?ar=16:9&fit=crop&crop=faces&fm=webp",
#             "https://d.newsweek.com/en/full/2603083/bill-maher.jpg?w=1600&h=1200&q=88&f=a39cb4f94b906da2034e9d784c6d2941",
#             "https://media.npr.org/assets/img/2023/09/14/gettyimages-866151214_wide-76eb9e6773132611d45af8bb32fa0dd16088d2a9.jpg?s=1100&c=50&f=jpeg",
#             "https://cdnph.upi.com/svc/sv/upi/9221730999107/2024/1/78852f5b1f6cde4303514e743e208fdd/Bill-Maher-comedy-special-Is-Anyone-Else-Seeing-This-coming-in-January.jpg",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2024/05/BillMaherGettyImages-2074809903.jpg?w=1296&h=730&crop=1",
#             "https://i.guim.co.uk/img/media/51c8821a9c1cfdbff33612063e293ba7432a591b/479_69_2515_1509/master/2515.jpg?width=465&dpr=1&s=none&crop=none",
#             "https://heroicmedia.org/wp-content/uploads/2024/04/bill-maher-scaled.webp",
#             "https://9b16f79ca967fd0708d1-2713572fef44aa49ec323e813b06d2d9.ssl.cf2.rackcdn.com/1140x_a10-7_cTC/author-and-comedian-bill-maher-1720109364.jpg",
#             "https://www.catholicleague.org/wp-content/uploads/2015/06/BillMaher.jpg",
#             "https://nypost.com/wp-content/uploads/sites/2/2025/03/bill-maher-100922151.jpg?quality=75&strip=all&w=1024",
#             "https://static01.nyt.com/images/2019/10/06/magazine/06mag-talk-01/06mag-talk-01-videoSixteenByNine3000.jpg?year=2019&h=1687&w=3000&s=4e4bafdd643f7c96d411c7f1de4960f633e3351a53371769c473d426c4b14084&k=ZQJBKqZ0VN&tw=1",
#             "https://cloudfront-us-east-1.images.arcpublishing.com/advancelocal/MWFO34OGWNBPZHG34KPFYANLU4.jpg",
#             "https://resizing.flixster.com/-XZAfHZM39UwaGJIFWKAE8fS0ak=/v3/t/assets/1076_v9_bb.jpg",
#             "https://static01.nyt.com/images/2023/09/18/business/18strike-maher/18strike-maher-mediumSquareAt3X.jpg",
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Bill_Maher_by_David_Shankbone.jpg/1200px-Bill_Maher_by_David_Shankbone.jpg",
#             "https://deadline.com/wp-content/uploads/2022/01/Bill-Maher-2021-HBO.jpeg",
#             "https://www.rclco.com/wp-content/uploads/2023/11/Bill-Maher-Headshot-2023-Cropped.jpg",
#             "https://blog-admin.siriusxm.com/wp-content/uploads/2024/05/bill-maher-megyn-kelly3.jpg?w=885",
#             "https://people.com/thmb/4cdiKi_Ay3Kh8nAwgmsQDK8bAuA=/4000x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(234x0:236x2)/bill_maher470-c3978280f1794e2eb96d507667dcd344.jpg",
#             "https://mediaassets.cbre.com/-/media/project/cbre/shared-site/headshots/b/i/l/bill-maher-752x.jpg",
#             "https://www.sky.com/assets2/real-time-with-bill-maher-panel-ccbacc2b.jpg",
#             "https://thehill.com/wp-content/uploads/sites/2/2024/06/Joe-Biden-Bill-Maher-06.29.png?strip=1",
#             "https://images.nymag.com/news/features/maher120416_1_560.jpg",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2020/04/real_time_with_bill_maher.jpg?w=1296&h=730&crop=1",
#             "https://s.yimg.com/ny/api/res/1.2/YNLqG3jcObJi7U7o5sY6eQ--/YXBwaWQ9aGlnaGxhbmRlcjt3PTEyNDI7aD0xNDM4/https://media.zenfs.com/en/us_magazine_896/5f6f661f95aa4d2389c8e09c8441d720",
#             "https://i.ytimg.com/vi/V7R9h-SCX68/sddefault.jpg",
#             "https://cloudfront-us-east-2.images.arcpublishing.com/reuters/U3SVXGO5VJJVRM2EG2S627APPQ.jpg",
#             "https://images.cdn.prd.api.discomax.com/2025/01/22/a4e4a418-64f1-3bbc-889e-64fda09e7fcb.jpeg?f=jpg&q=75&w=1280",
#             "https://img.etimg.com/thumb/width-420,height-315,imgsize-3956,resizemode-75,msid-114404047/news/international/us/heres-why-bill-maher-desperately-wants-kamala-harris-to-win-the-us-elections-2024.jpg",
#             "https://www.thenation.com/wp-content/uploads/2015/03/bill_maher_ap_img.jpg",
#             "https://deadline.com/wp-content/uploads/2021/05/Real-Time-with-Bill-Maher-2021.jpg?crop=54px%2C69px%2C1800px%2C1009px&resize=681%2C383",
#             "https://cdn.britannica.com/27/157827-050-F7F43E50/Bill-Maher-2011.jpg?w=400&h=300&c=crop",
#             "https://ew.com/thmb/98MJix5zx5yyE145IXgEJT0LaaM=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/real-time-with-bill-maher-3-2000-9f7cf5d3098d462785975c9c8380fd75.jpg",
#             "https://media.vanityfair.com/photos/63d7e820269be31f5c11ceda/16:9/w_2000,h_1125,c_limit/Bill-Maher.jpg",
#             "https://www.politico.com/dims4/default/fbf2ff9/2147483647/legacy_thumbnail/1200x800%3E/quality/90/?url=http%3A%2F%2Fs3-origin-images.politico.com%2F2012%2F03%2F120316_bill_maher_facts_605_ap.jpg",
#             "https://static.independent.co.uk/2024/05/06/22/newFile-11.jpg",
#             "https://www.nydailynews.com/wp-content/uploads/migration/2009/10/26/6ZB6ZJIAZAWW7T62PFIOAWAJGQ.jpg?w=535",
#             "https://opinion-images.wsj.net/im-86981325/?size=1.5",
#             "https://people.com/thmb/gP3f4pY-s7GWQQPNZEJsTEEhrP4=/4000x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(781x309:783x311)/Bill-Maher-011225-2-71f441edef3c498bb5657f9fb7814089.jpg",
#             "https://www.onthisday.com/images/people/bill-maher.jpg?w=360",
#             "https://deadline.com/wp-content/uploads/2025/03/bill-maher-real-time-taylor-swift.jpg",
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/BillMaherSept10.jpg/1200px-BillMaherSept10.jpg",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2024/05/1200/675/BillMaher53124.jpg?ve=1&tl=1",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2025/03/1200/675/mahermuskchainsaw2.jpg?ve=1&tl=1",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2022/05/1200/675/d849cd06-Maher.jpg?ve=1&tl=1",
#             "https://thecomedystore.com/wp-content/uploads/2023/06/e5984e4c255ee9ceee6f7c2fde95508a.jpg",
#             "https://edge.dailyvoice.com/upload/FotoJet_HnzrXOO.jpg?width=720",
#             "https://static.hbo.com/2022-04/bill-maher-adulting-ka-1920.jpg?w=1200",
#             "https://m.media-amazon.com/images/M/MV5BNTQ4MzgwMTcxMl5BMl5BanBnXkFtZTcwNjAyNzQzMg@@._V1_FMjpg_UX1000_.jpg",
#             "https://cdn.choosechicago.com/uploads/2024/10/Bill-Maher-BIC-1280x820-1.jpg",
#             "https://a57.foxnews.com/cf-images.us-east-1.prod.boltdns.net/v1/static/694940094001/d31c31e9-a62d-4573-bd77-e50bb9e4fa63/236034eb-ca00-4948-a6d3-f80df7f2420b/1280x720/match/896/500/image.jpg?ve=1&tl=1",
#             "https://latenighter.com/wp-content/uploads/2024/07/TSDSARA_EC006.jpg",
#             "https://media.gq.com/photos/5c7571166a6e6372b18b6e67/16:9/w_2560%2Cc_limit/Bill-Maher-Retire-Bitch-GQ-2019-022619.jpg",
#             "https://www.nj.com/resizer/v2/N72EFUO2I5A5PKGQ4WNP55CRJU.jpg?auth=4c2b5eb7ee833d3e224fb747b26cc422ee5cf8f00af6479f1cfae7380dcc2b5f&width=500&quality=90",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2025/03/dana-carvey-bill-maher.jpg?w=1296&h=730&crop=1",
#             "https://www.thekeyexecutives.com/wp-content/uploads/2021/08/Bill-Maher.jpg",
#             "https://ffrf.org/wp-content/uploads/1980/01/90c519c0607749e3c2d669154a64428f.jpg",
#             "https://people.com/thmb/QgC19bpHBlHGR-CodUM6tV1T5hQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(999x0:1001x2)/bill-maher-1d579295ab1e4fa7b361b5dca2627685.jpg",
#             "https://variety.com/wp-content/uploads/2024/11/GettyImages-2153895970-e1730996650378.jpg?w=1000&h=667&crop=1",
#             "https://kleinhansbuffalo.org/wp-content/uploads/sites/2/2022/03/Bill-Maher-credit-David-Becker-WireImage-11.25.13-scaled.jpg",
#             "https://nypost.com/wp-content/uploads/sites/2/2025/03/bill-maher-visits-megyn-kelly-100868663.jpg?quality=75&strip=all&w=1200",
#             "https://deadline.com/wp-content/uploads/2025/03/Bill-Maher-PA-Gov.-Josh-Shapiro.jpg",
#             "https://media.zenfs.com/en/parade_250/9d6e4237175a3d23199b5a3c8499c057",
#             "https://static.hbo.com/content/dam/hbodata/series/real-time-with-bill-maher/episodes/15/800603_RTWBM_S15_L0_3000x1688.jpg",
#             "https://m.media-amazon.com/images/S/pv-target-images/c6b83ca34a7c8ddd424a0ccb97d2cfc1bec750906d05f50ac4666dd7a7daf3d2._SX1080_FMjpg_.jpg",
#             "https://cbsaustin.com/resources/media2/16x9/full/1015/center/80/09186cb0-09e0-4fb3-9b25-5f9347afec0e-large16x9_AP21134082736586.jpg",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2024/03/Bill-Maher-getty-H-2024.jpg?w=1296&h=730&crop=1",
#             "https://external-preview.redd.it/its-an-honor-bill-maher-explains-why-hes-going-to-meet-soon-v0-s3QJw9zggIIDHmroTqYVCLKppOMeZGTkVvIqoChe7cs.jpg?width=640&crop=smart&auto=webp&s=1f860ffe64573313f84cdeb5e8669d3f11ef7860",
#             "https://people.com/thmb/CSefpKe40XHzdl7qwhpoQyWtqB4=/4000x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(535x0:537x2)/bill-maher--Noor-Alfallah-1-100924-a7a0c0ab3b4f46dd8606832a1990eb63.jpg",
#             "https://www.apbspeakers.com/media/cirhnbev/maher_bill.jpg?rxy=0.40666666666666668,0.38016528925619836&height=775&quality=70&v=1daf95534af6fe0)",
#             "https://static.cdn.turner.com/styles/header_image_1500x500_cropped/s3/2025-01/realtimewithbillmaher_s23_ho_ka_27x40_publicity_0.jpg?h=cfa352ee&itok=U7X-ziHP",
#             "https://variety.com/wp-content/uploads/2017/01/real-time-with-bill-maher.jpg?w=1000&h=562&crop=1",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2024/04/1200/675/MaherTrump.jpg?ve=1&tl=1"
#           ]
#         },
#         {
#           "characterName": "Donald Trump",
#           "images": [
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/TrumpPortrait.jpg/1200px-TrumpPortrait.jpg",
#             "https://bidenwhitehouse.archives.gov/wp-content/uploads/2021/01/45_donald_trump_w-1250.jpg",
#             "https://cdn.britannica.com/88/190788-050-4F6F73F2/Donald-Trump-2016.jpg",
#             "https://hips.hearstapps.com/hmg-prod/images/gettyimages-2194420718-67d9b4e326598.jpg?crop=0.646xw:0.969xh;0.148xw,0&resize=640:*",
#             "https://www.aljazeera.com/wp-content/uploads/2025/03/AP25073719749030-1741984709.jpg?resize=770%2C513&quality=80",
#             "https://i.natgeofe.com/k/5e4ea67e-2219-4de4-9240-2992faef0cb6/trump-portrait.jpg",
#             "https://dims.apnews.com/dims4/default/8dba174/2147483647/strip/true/crop/5337x3563+0+0/resize/599x400!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F2b%2F65%2Fc1e9e86ec37b84b09b836148ea93%2Fba91ce472e28494693e16566e9a6fdd4",
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Donald_Trump_%288567815294%29_%28cropped%29.jpg/640px-Donald_Trump_%288567815294%29_%28cropped%29.jpg",
#             "https://i.abcnewsfe.com/a/030dec16-260c-4be0-8fe6-28b41c06bb36/donald-trump-9-gty-gmh-250313_1741885130014_hpMain.jpg",
#             "https://cdn.britannica.com/21/197021-050-D0AC3E38/portrait-Donald-Trump.jpg",
#             "https://c.files.bbci.co.uk/F332/production/_133585226_us24_flashcards_trump.png",
#             "https://people.com/thmb/1wrT7EDuoarBvk1MOgijDFM2AsE=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(999x0:1001x2)/barron-trump-donald-trump-inauguration-012025-3bf63b80cbe147388f5419c4a1f453f3.jpg",
#             "https://media.newyorker.com/photos/669324fd14c38da9ac7b7815/master/w_2560%2Cc_limit/Trump-shot-AP24195851874181.jpg",
#             "https://media-cldnry.s-nbcnews.com/image/upload/t_fit-560w,f_auto,q_auto:best/rockcms/2024-11/241108-donald-trump-al-0936-52ff7b.jpg",
#             "https://live-production.wcms.abc-cdn.net.au/48743062bb31e6708bdc6716e5a65628?impolicy=wcms_crop_resize&cropH=1316&cropW=2340&xPos=0&yPos=112&width=862&height=485",
#             "https://ru.usembassy.gov/wp-content/uploads/sites/138/AP25020618863748-1140x684.jpg",
#             "https://static01.nyt.com/newsgraphics/2019/08/01/candidate-pages/3b31eab6a3fd70444f76f133924ae4317567b2b5/trump.jpg",
#             "https://www.aljazeera.com/wp-content/uploads/2025/03/AP25079748045798-1742504198.jpg?resize=770%2C513&quality=80",
#             "https://dobetter.esade.edu/sites/default/files/post/2024-10/victory-trump-EU-m.webp",
#             "https://m.media-amazon.com/images/I/81afCibN3PL.jpg",
#             "https://fortune.com/img-assets/wp-content/uploads/2016/04/tru05_a.jpg",
#             "https://i.abcnewsfe.com/a/6f82c457-4ae9-42bd-a9d6-dd430dabce19/donald-trump-10-ap-gmh-241106_1730906289206_hpMain_16x9.jpg?w=992",
#             "https://static01.nyt.com/images/2024/12/11/multimedia/11trump-news-time2-zpgt/11trump-news-time2-zpgt-mediumSquareAt3X.jpg",
#             "https://dims.apnews.com/dims4/default/c34ef7c/2147483647/strip/true/crop/4934x3294+0+0/resize/599x400!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fe7%2F74%2F2168cbc28168413d8179c6267766%2F992c8bdfe60643768471321b16c9ba98",
#             "https://m.media-amazon.com/images/I/61AobmicMsL.jpg",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2024/11/dt-winner-1024x585.jpg",
#             "https://fortune.com/img-assets/wp-content/uploads/2025/02/GettyImages-2201327019-e1740324709905.jpg?w=1440&q=75",
#             "https://i.insider.com/6793e777e7e163809a2279ba?width=700",
#             "https://www.bu.edu/files/2025/01/trump-100-days-feat.jpg",
#             "https://snworksceo.imgix.net/dtc/df21c545-b4b8-4d85-ae96-507181573926.sized-1000x1000.jpg?w=1000",
#             "https://npr.brightspotcdn.com/dims3/default/strip/false/crop/4890x3260+0+0/resize/1100/quality/50/format/jpeg/?url=http%3A%2F%2Fnpr-brightspot.s3.amazonaws.com%2Feb%2F26%2F3424469c4d29b63b0b2683c3a4f2%2Fap24161743821924.jpg",
#             "https://cdn.britannica.com/45/193845-050-A7905463/Donald-Trump-front-Tower-New-York-City-August-2008.jpg",
#             "https://media-cldnry.s-nbcnews.com/image/upload/t_fit-760w,f_auto,q_auto:best/rockcms/2025-02/250219-miami-beach-donald-trump-ac-935p-0b75a6.jpg",
#             "https://cdn.theatlantic.com/thumbor/YkmuUicCffTSqDzT6wL1DdyQxYo=/0x416:3712x2504/960x540/media/img/mt/2019/09/RTS2PCAV/original.jpg",
#             "https://deadline.com/wp-content/uploads/2019/05/donald-trump-the-apprentice.jpg?w=1000",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2024/11/GettyImages-2183216502-1024x683.jpg",
#             "https://media.vanityfair.com/photos/677e9d5ac1ae5d57ab0d79b9/master/w_2560%2Cc_limit/2192852996",
#             "https://cdn.abcotvs.com/dip/images/15515767_110524-cc-ap-trump-pointing-img.jpg",
#             "https://www.aljazeera.com/wp-content/uploads/2024/12/2024-12-07T175425Z_1928241406_RC2FKBARY8Z2_RTRMADP_3_USA-TRUMP-FRANCE-1733734296.jpg?resize=770%2C513&quality=80",
#             "http://media.newyorker.com/photos/64b6d2167ac8745598305585/master/pass/Chotiner-Trump-Power.jpg",
#             "https://images.axios.com/B7ruxTmk9XUpH1NmSzPCSvUAXeg=/0x321:7008x4263/1920x1080/2024/04/30/1714481388445.jpg?w=3840",
#             "https://platform.vox.com/wp-content/uploads/sites/2/chorus/uploads/chorus_asset/file/7448639/cover.png?quality=90&strip=all&crop=0,7.1190476190476,100,85.761904761905",
#             "https://www.economist.com/cdn-cgi/image/width=1424,quality=80,format=auto/content-assets/images/20240706_USP503.jpg",
#             "https://si.usembassy.gov/wp-content/uploads/sites/113/2025/02/trump_profile_smithsonian-.jpeg",
#             "https://dims.apnews.com/dims4/default/e13ae1d/2147483647/strip/true/crop/3742x2495+0+0/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fb1%2Fd8%2F88d8271784c02ce9c32af6ae280f%2Fb815da18305c4c83904f8ecebacd7550",
#             "https://www.investopedia.com/thmb/kf9uXBLfpLnbZKZXkH6v0hmwZas=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/DonaldTrump-977a6dad59e34debb7a6ab3c92328510.jpg",
#             "https://cdn.cfr.org/sites/default/files/styles/slide_3_2/public/image/2024/06/Trump_final_1.jpg.webp",
#             "https://whyy.org/wp-content/uploads/2025/01/donald-trump-2025-01-08.jpg",
#             "https://media.cnn.com/api/v1/images/stellar/prod/240119105200-donald-trump-0117.jpg?c=16x9&q=h_833,w_1480,c_fill",
#             "https://imageio.forbes.com/specials-images/imageserve/615db66e82a521d2906bb8e6/Donald-Trump/1960x0.jpg?format=jpg&width=960",
#             "https://static.foxnews.com/foxnews.com/content/uploads/2024/11/donald-trump-campaign-msg.jpg",
#             "https://c.files.bbci.co.uk/4616/live/d710a6a0-d4b0-11ef-9fd6-0be88a764111.jpg",
#             "https://media-cldnry.s-nbcnews.com/image/upload/t_fit-560w,f_auto,q_auto:best/rockcms/2025-02/250220-donald-trump-ac-1116p-d87512.jpg",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2025/02/2025-02-24T174606Z_1203290860_RC251DAHQ3D9_RTRMADP_3_USA-FRANCE-1024x682.jpg",
#             "https://whyy.org/wp-content/uploads/2025/03/trump-mccormick-ap-032325-768x512.jpg",
#             "https://www.politico.com/dims4/default/31b4beb/2147483647/strip/true/crop/3688x2459+0+0/resize/630x420!/quality/90/?url=https%3A%2F%2Fstatic.politico.com%2F3e%2F3f%2F8cc047fd4f6d900edaee705fcbd9%2Felection-2024-trump-43634.jpg",
#             "https://imagenes.elpais.com/resizer/v2/2PPMQOADCKQOGKLO3KFGNDPYMI.jpg?auth=e5741ec89d81b21a0c8c1cd67f4b68dc4f2c46482afbb5a07f007a1290ce6329&width=1960&height=1470&smart=true",
#             "https://i.abcnewsfe.com/a/25e5fdcf-6e28-41f5-a39d-969d1d8b34cd/inauguration-48-gty-gmh-250120_1737393470475_hpMain.jpg",
#             "https://assets.bwbx.io/images/users/iqjWHBFdfxIU/in5580yMORDI/v1/-1x-1.webp",
#             "https://s.france24.com/media/display/36ac61e6-00dd-11f0-8e1a-005056bf30b7/w:1280/p:16x9/063_2204326270.jpg",
#             "https://images.seattletimes.com/wp-content/uploads/2024/04/Y6TwKOO3.jpg?d=780x520",
#             "https://fortune.com/img-assets/wp-content/uploads/2025/01/GettyImages-2191378216_bf5b91-e1736185018887.jpg?w=1440&q=75",
#             "https://media.cnn.com/api/v1/images/stellar/prod/gettyimages-2174482169-copy.jpg?c=16x9&q=h_833,w_1480,c_fill",
#             "https://npr.brightspotcdn.com/dims3/default/strip/false/crop/1024x683+0+0/resize/1100/quality/50/format/jpeg/?url=http%3A%2F%2Fnpr-brightspot.s3.amazonaws.com%2F5d%2F65%2F6ab062cc4b2495b795e13556c4bf%2Fgettyimages-2194985045.jpg",
#             "https://static01.nyt.com/images/2024/11/05/multimedia/pol-trump-wins-hfo-new0101-qzgw/pol-trump-wins-hfo-new0101-qzgw-mediumSquareAt3X.jpg",
#             "https://www.reuters.com/resizer/v2/65NBG2ZKS5IJJMJ4CTFYINSKRQ.jpg?auth=aa285593f752219d97af0b7693fd83892389623c6c7aeeabf471431f44d10a5f&width=7049&quality=80",
#             "https://media.newyorker.com/photos/671abac398e2f88da89a9d60/master/w_2560%2Cc_limit/Glasser-TrumpView.jpg",
#             "https://media.nbcphiladelphia.com/2025/02/web-250209-donald-trump-getty.jpg?quality=85&strip=all&resize=1200%2C675",
#             "https://cdn.theatlantic.com/thumbor/5o08pfdzYI358Bx27p2MobrBek8=/0x0:3000x3750/648x810/media/img/2024/11/07/trump_normalize/original.jpg",
#             "https://dims.apnews.com/dims4/default/34eb7d8/2147483647/strip/true/crop/2143x1429+0+0/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F86%2F6d%2F7729139dafde91820e1461acfce6%2Fce5cad3e4a514283b331e2aa89d7eb85",
#             "https://thumbnails.texastribune.org/k_5DHToac7q13Yb1mLFbOlGSNZY=/850x570/smart/filters:quality(75)/https://static.texastribune.org/media/files/210b3a66a97e03e7ca7acd2349408982/Trump%20and%20Melania%20REUTERS%20TT.jpg",
#             "https://lailluminator.com/wp-content/uploads/2024/02/TrumpSC-scaled-1-1536x1024.jpg",
#             "https://live-production.wcms.abc-cdn.net.au/09cc8f05ffa54eac464fb108453b4a58?impolicy=wcms_crop_resize&cropH=2813&cropW=5000&xPos=0&yPos=260&width=862&height=485",
#             "https://images.wsj.net/im-229976?size=1.5&width=640",
#             "https://www.hrw.org/sites/default/files/styles/embed_xxl/public/media_2025/01/202501usp_usa_donald_trump_executive_orders.jpg?itok=Zq-sUSVK",
#             "https://i.abcnewsfe.com/a/0f6aa83e-b2e3-4d9d-bae6-681d4a8585f3/apprentice-2-rt-er-250219_1740001709973_hpMain_9x16.jpg?w=992",
#             "https://deadline.com/wp-content/uploads/2025/02/madonna-donald-trump.jpg?w=681&h=383&crop=1",
#             "https://image.sciencenorway.no/2478477.webp?imageId=2478477&x=0.00&y=0.00&cropw=100.00&croph=100.00&width=960&height=644&format=jpg",
#             "https://www.bu.edu/files/2024/11/crop-AP24332089848604.jpg",
#             "https://ichef.bbci.co.uk/news/480/cpsprodpb/7304/live/22147890-ff9d-11ef-b0ab-6d56b02ced8f.jpg.webp",
#             "https://i.kinja-img.com/image/upload/c_fit,q_60,w_645/99b4fd6fc475aab45b3e25c97cc8eb3d.jpg",
#             "https://img.lemde.fr/2025/01/22/0/0/5435/3624/664/0/75/0/7ebe576_sirius-fs-upload-1-vdgaio8mmx2w-1737535305564-049554.jpg",
#             "https://images.theconversation.com/files/571418/original/file-20240125-27-a3g4ec.jpg?ixlib=rb-4.1.0&rect=0%2C0%2C2907%2C1926&q=20&auto=format&w=320&fit=clip&dpr=2&usm=12&cs=strip",
#             "https://www.americanprogress.org/wp-content/uploads/sites/2/2025/02/GettyImages-2201118913cropped.jpg",
#             "https://platform.vox.com/wp-content/uploads/sites/2/2024/07/GettyImages-2161922012.jpg?quality=90&strip=all&crop=0.021331058020472%2C0%2C99.957337883959%2C100&w=2400",
#             "https://www.usnews.com/dims4/USNEWS/3516ede/2147483647/thumbnail/970x647/quality/85/?url=https%3A%2F%2Fwww.usnews.com%2Fcmsmedia%2Fb4%2Ff9%2F0835906340ffbb3196a84f58ae60%2F170921-donaldtrump10things-editorial.jpg",
#             "https://cdn.britannica.com/31/149831-050-83A0E45B/Donald-J-Trump-2010.jpg",
#             "https://www.thenation.com/wp-content/uploads/2024/07/Donald-Trump-Department-of-Education.jpg",
#             "https://www.citizensforethics.org/wp-content/uploads/2021/02/46451130962_d151b497d6_b-1200x675.jpg",
#             "https://www.thoughtco.com/thmb/OIQxajVv1AMskdgr21fgM-qpCOo=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-1163026796-ef8fdfa1a68a471d8bda5aeab73c98c0.jpg",
#             "https://people.com/thmb/-Z24JV9E4vfl2olDngYIcN8JLdc=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(999x0:1001x2)/donald-trump-melania-121224-3da0aec3465f4588b25d7217926c431b.jpg"
#           ]
#         }
#       ],
#       "clips": [
#         {
#           "characterName": "Clip",
#           "videos": [
#             {
#               "url": "https://www.youtube.com/watch?v=c9txhe-5EJA",
#               "vidclips": [
#                 {
#                   "endTime": "00:00:03.37",
#                   "mute": True,
#                   "startTime": "00:00:02.75"
#                 }
#               ]
#             }
#           ]
#         },
#         {
#           "characterName": "Bill Maher",
#           "videos": [
#             {
#               "url": "https://youtu.be/vRz47H4iFv4",
#               "vidclips": [
#                 {
#                   "endTime": "00:02:37.45",
#                   "mute": False,
#                   "startTime": "00:01:50.06",
#                   "tag": "billions"
#                 },
#                 {
#                   "endTime": "00:03:34.37",
#                   "mute": False,
#                   "startTime": "00:02:45.87",
#                   "tag": "biology"
#                 }
#               ]
#             },
#             {
#               "url": "https://www.youtube.com/watch?v=c9txhe-5EJA",
#               "vidclips": [
#                 {
#                   "endTime": "00:00:22.65",
#                   "mute": True,
#                   "startTime": "00:00:16.87"
#                 }
#               ]
#             },
#             {
#               "url": "https://www.youtube.com/watch?v=ZkFUU-xJM6I",
#               "vidclips": [
#                 {
#                   "endTime": "00:00:14.27",
#                   "mute": True,
#                   "startTime": "00:00:06.08"
#                 }
#               ]
#             },
#             {
#               "url": "https://www.youtube.com/watch?v=fMF0bser1aM",
#               "vidclips": [
#                 {
#                   "endTime": "00:01:26.95",
#                   "mute": True,
#                   "startTime": "00:01:18.75"
#                 }
#               ]
#             },
#             {
#               "url": "https://www.youtube.com/watch?v=63KXfwC9BdU",
#               "vidclips": [
#                 {
#                   "endTime": "00:02:01.58",
#                   "mute": True,
#                   "startTime": "00:01:58.30"
#                 },
#                 {
#                   "endTime": "00:02:07.30",
#                   "mute": True,
#                   "startTime": "00:02:03.17"
#                 },
#                 {
#                   "endTime": "00:02:21.10",
#                   "mute": True,
#                   "startTime": "00:02:17.20"
#                 },
#                 {
#                   "endTime": "00:02:27.25",
#                   "mute": True,
#                   "startTime": "00:02:23.30"
#                 }
#               ]
#             }
#           ]
#         },
#         {
#           "characterName": "Donald Trump",
#           "videos": [
#             {
#               "url": "https://www.youtube.com/watch?v=0fFezDZ1jaM",
#               "vidclips": [
#                 {
#                   "endTime": "00:01:06.81",
#                   "mute": True,
#                   "startTime": "00:01:00.10"
#                 },
#                 {
#                   "endTime": "00:01:12.96",
#                   "mute": True,
#                   "startTime": "00:01:07.72"
#                 }
#               ]
#             },
#             {
#               "url": "https://www.youtube.com/watch?v=SwDZ83uBsK4",
#               "vidclips": [
#                 {
#                   "endTime": "00:00:07.10",
#                   "mute": True,
#                   "startTime": "00:00:02.01"
#                 },
#                 {
#                   "endTime": "00:00:12.33",
#                   "mute": True,
#                   "startTime": "00:00:07.85"
#                 },
#                 {
#                   "endTime": "00:00:21.16",
#                   "mute": True,
#                   "startTime": "00:00:14.36"
#                 }
#               ]
#             }
#           ]
#         }
#       ],
#       "section_characters": [
#         "Bill Maher",
#         "Donald Trump"
#       ],
#       "segment": [
#         {
#           "character": "Bill Maher",
#           "tag": "intro",
#           "text": "[CLIP: Sports by sex doesn't make sense. Yes, it does Because again, we haven't reinvented homo sapiens since crystal Pepsi came out I've spent three decades on TV mocking Republicans who said climate change was just a theory and now I got to deal with people who Say, you know, what else is just a theory by a but that can't happen overnight Because the people who aren't getting the food or the medicine they're the ones who are gonna suffer You don't see why people find this incredibly cruel You said earlier that we're treating it as if it's a it's an overweight person putting them on a crash diet oh, no, the federal government is essentially somebody who's 800 pounds and You cannot survive like this So yes There are some drastic measures that have to be taken and I will add this to you If you think it's bad now if we get into a real debt crisis It will be significantly worse for the American people and other people that we might help around the globe the best thing to do right now Getting quick. You're gonna have to shut some things down triage the area rebuild the stuff That is good that we all can agree with the other stuff. That's hyper political or makes no sense or fraudulent What about the Department of Education now? I I don't know that much about it, but I've never read good things Rahm Emanuel who I agree with almost everything here was a quote He said a third of eighth graders can't read and now he wants to close the department ]"
#         },
#         {
#           "character": "Donald Trump",
#           "text": "Okay everyone, let's dive in!"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "You know Bill Maher, correct?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Lately, he's really shaking things up, and honestly, it's the buzz of the town—just thinking about it could send your temper through the roof."
#         },
#         {
#           "character": "Donald Trump",
#           "text": "That famed funnyman and star of his own HBO show has made a daring move that truly blindsided us all—he’s actually defending Donald Trump!"
#         },
#         {
#           "character": "Bill Maher",
#           "tag": "billions",
#           "text": "[CLIP: If I give you a billion dollars and you can't tell me what happened to it, that to me is wasteful. That means you are not responsible. But if you can't tell me where it went, then what am I supposed to think? An $850 billion budget to an organization that can't pass an audit and tell you where that money went, like, I think most people would consider that somewhere in the realm of waste, fraud, or abuse because they would wonder why that money isn't well accounted for. But I'm a human being who lives on the earth and can't figure out how $850 billion to a department means that the rank and file still have to be on food stamps. Like to me, that's fucking corruption. ]"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "And can you believe this?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "After all the mocking he's done over the years, that's the guy."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Maher is really going for it, hitting hard at both political parties and he's not holding back at all."
#         },
#         {
#           "character": "Bill Maher",
#           "tag": "biology",
#           "text": "[CLIP: With communists, that human was no longer selfish. In America today, that human is no longer born male or female. And obesity is not something that affects health. You can be healthy at any size. Really, we voted on it. A formerly serious magazine last year published with a straight face an article called, Separating Sports by Sex Doesn't Make Sense. Yes, it does. Because again, we haven't reinvented homo sapien since crystal Pepsi came out. I've spent three decades on TV mocking Republicans who said climate change was just a theory. And now I've got to deal with people who say, you know what else is just a theory? ]"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "In the current heated political scene, his recent actions regarding burning issues such as education and immigration have polarized his audience—some are applauding, while others are absolutely livid!"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Hold up!"
#         },
#         {
#           "character": "Donald Trump",
#           "text": "If you're into content that pushes boundaries and keeps it authentic, don't forget to hit that subscribe button."
#         },
#         {
#           "character": "Donald Trump",
#           "text": "Go ahead, I’ll wait... Done?"
#         },
#         {
#           "character": "Donald Trump",
#           "text": "Great to have you!"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Now, let’s dive back into the discussion."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Bill Maher's Views Okay, let's delve into... all the juicy details of Bill Maher’s straightforward opinions, okay?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Lately, Maher has used his platform to expose his thoughts on some of today’s most pressing social issues."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "And let me tell you, he does not mince his words."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "His forthright analysis has been strikingly forthright on topics like the Department of Education, immigration laws in California, and let’s not forget those sensitive debates about gender in sports."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Now, let's discuss Bill's perspective on education."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "How do I phrase this..?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "For ages, education has been regarded almost as a holy, untouchable domain, right?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "But here's Maher, calling out issues, pointing fingers, and declaring, \"Something here just doesn’t make sense.\" Frankly, his criticism isn’t without basis—it’s supported by data that can’t be overlooked."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Take this alarming statistic, for instance:"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "a third of our nation’s eighth graders aren’t reading at the appropriate grade level."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "So, there's Maher, virtually yelling, \"What's actually happening here?\" Honestly, are we just sinking billions into an educational system that’s deteriorating year after year?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "According to Bill, it seems the Department of Education is more of a financial sinkhole than a proponent of education, sort of like a vortex where taxpayer money vanishes."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Immigration Policies Moving on to Maher on immigration—here’s where it heats up even more."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "You see, lawmakers in California have chosen to offer government aid to undocumented immigrants for home purchases."
#         },
#         {
#           "character": "Bill Maher",
#           "tag": "lawmakers",
#           "text": "[CLIP: The California lawmakers just passed a law giving government assistance to undocumented immigrants to buy houses. That's kind of a different place than the Democratic Party used to be on immigration, is it not? Well, let me just say, immigration had always been a bipartisan issue. I refer you to the- But not free houses. Well, that's not free housing. It's the American dream, being available to more people. California's always in the lead. Maybe others will follow that lead, but that's up to those states. So you'd vote for this law? I'm not familiar with exactly what it is, but making the American dream of home ownership available to all people is something we have to do for people who are here now. This is for the undocumented. ]"
#         }
#       ],
#       "section_images": [
#         {
#           "character": "Bill Maher",
#           "images": [
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/BillMaherSept10.jpg/1200px-BillMaherSept10.jpg",
#             "https://static.hbo.com/2024-12/real-time-with-bill-maher-s23-ka-1920.jpg",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2024/05/1200/675/BillMaher53124.jpg?ve=1&tl=1",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2025/03/1200/675/mahermuskchainsaw2.jpg?ve=1&tl=1",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2022/05/1200/675/d849cd06-Maher.jpg?ve=1&tl=1",
#             "https://thecomedystore.com/wp-content/uploads/2023/06/e5984e4c255ee9ceee6f7c2fde95508a.jpg",
#             "https://edge.dailyvoice.com/upload/FotoJet_HnzrXOO.jpg?width=720",
#             "https://m.media-amazon.com/images/M/MV5BNTQ4MzgwMTcxMl5BMl5BanBnXkFtZTcwNjAyNzQzMg@@._V1_FMjpg_UX1000_.jpg",
#             "https://static.hbo.com/2022-04/bill-maher-adulting-ka-1920.jpg?w=1200",
#             "https://cdn.choosechicago.com/uploads/2024/10/Bill-Maher-BIC-1280x820-1.jpg",
#             "https://www.nj.com/resizer/v2/N72EFUO2I5A5PKGQ4WNP55CRJU.jpg?auth=4c2b5eb7ee833d3e224fb747b26cc422ee5cf8f00af6479f1cfae7380dcc2b5f&width=500&quality=90",
#             "https://media.gq.com/photos/5c7571166a6e6372b18b6e67/16:9/w_2560%2Cc_limit/Bill-Maher-Retire-Bitch-GQ-2019-022619.jpg",
#             "https://variety.com/wp-content/uploads/2025/03/don.png?w=1000&h=667&crop=1",
#             "https://latenighter.com/wp-content/uploads/2024/07/TSDSARA_EC006.jpg",
#             "https://a57.foxnews.com/cf-images.us-east-1.prod.boltdns.net/v1/static/694940094001/d31c31e9-a62d-4573-bd77-e50bb9e4fa63/236034eb-ca00-4948-a6d3-f80df7f2420b/1280x720/match/896/500/image.jpg?ve=1&tl=1",
#             "https://i0.wp.com/www.iconvsicon.com/wp-content/uploads/2025/01/billmaher2.webp?resize=800%2C1185&ssl=1",
#             "https://static.wikia.nocookie.net/usam/images/9/99/Bill_Maher%27s_Official_Portrait.png/revision/latest/scale-to-width-down/1170?cb=20221225200507",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2025/03/dana-carvey-bill-maher.jpg?w=1296&h=730&crop=1",
#             "https://www.thekeyexecutives.com/wp-content/uploads/2021/08/Bill-Maher.jpg",
#             "https://ffrf.org/wp-content/uploads/1980/01/90c519c0607749e3c2d669154a64428f.jpg",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2024/04/1200/675/MaherTrump.jpg?ve=1&tl=1",
#             "https://static.cdn.turner.com/styles/header_image_1500x500_cropped/s3/2025-01/realtimewithbillmaher_s23_ho_ka_27x40_publicity_0.jpg?h=cfa352ee&itok=U7X-ziHP",
#             "https://variety.com/wp-content/uploads/2017/01/real-time-with-bill-maher.jpg?w=1000&h=562&crop=1",
#             "https://www.apbspeakers.com/media/cirhnbev/maher_bill.jpg?rxy=0.40666666666666668,0.38016528925619836&height=775&quality=70&v=1daf95534af6fe0)",
#             "https://people.com/thmb/CSefpKe40XHzdl7qwhpoQyWtqB4=/4000x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(535x0:537x2)/bill-maher--Noor-Alfallah-1-100924-a7a0c0ab3b4f46dd8606832a1990eb63.jpg",
#             "https://external-preview.redd.it/its-an-honor-bill-maher-explains-why-hes-going-to-meet-soon-v0-s3QJw9zggIIDHmroTqYVCLKppOMeZGTkVvIqoChe7cs.jpg?width=640&crop=smart&auto=webp&s=1f860ffe64573313f84cdeb5e8669d3f11ef7860",
#             "https://ntvb.tmsimg.com/assets/p185450_b_h8_ab.jpg?w=1280&h=720",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2024/03/Bill-Maher-getty-H-2024.jpg?w=1296&h=730&crop=1",
#             "https://cbsaustin.com/resources/media2/16x9/full/1015/center/80/09186cb0-09e0-4fb3-9b25-5f9347afec0e-large16x9_AP21134082736586.jpg",
#             "https://m.media-amazon.com/images/S/pv-target-images/c6b83ca34a7c8ddd424a0ccb97d2cfc1bec750906d05f50ac4666dd7a7daf3d2._SX1080_FMjpg_.jpg",
#             "https://deadline.com/wp-content/uploads/2025/03/Bill-Maher-PA-Gov.-Josh-Shapiro.jpg",
#             "https://lookaside.instagram.com/seo/google_widget/crawler/?media_id=3594122346437716064",
#             "https://static.hbo.com/content/dam/hbodata/series/real-time-with-bill-maher/episodes/15/800603_RTWBM_S15_L0_3000x1688.jpg",
#             "https://www.billmaher.com/wp-content/uploads/2019/04/Web-2025-Banner.jpg",
#             "https://media.zenfs.com/en/parade_250/9d6e4237175a3d23199b5a3c8499c057",
#             "https://nypost.com/wp-content/uploads/sites/2/2025/03/bill-maher-visits-megyn-kelly-100868663.jpg?quality=75&strip=all&w=1200",
#             "https://kleinhansbuffalo.org/wp-content/uploads/sites/2/2022/03/Bill-Maher-credit-David-Becker-WireImage-11.25.13-scaled.jpg",
#             "https://i.ytimg.com/vi/d6qhapDVUNc/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLDufefHZ79MVhYlabSFMw_gQP8dFA",
#             "https://people.com/thmb/QgC19bpHBlHGR-CodUM6tV1T5hQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(999x0:1001x2)/bill-maher-1d579295ab1e4fa7b361b5dca2627685.jpg",
#             "https://variety.com/wp-content/uploads/2024/11/GettyImages-2153895970-e1730996650378.jpg?w=1000&h=667&crop=1",
#             "https://dw-wp-production.imgix.net/2025/03/GettyImages-2202961976-scaled.jpg?ar=16:9&fit=crop&crop=faces&fm=webp",
#             "https://i0.wp.com/www.nydailynews.com/wp-content/uploads/2024/10/241022_maher2.jpg?fit=620%2C9999px&ssl=1",
#             "https://m.media-amazon.com/images/M/MV5BMTUxNTY4NzIwMl5BMl5BanBnXkFtZTgwNDIxMTQ5NjE@._V1_.jpg",
#             "https://d.newsweek.com/en/full/2603083/bill-maher.jpg?w=1600&h=1200&q=88&f=a39cb4f94b906da2034e9d784c6d2941",
#             "https://i.ytimg.com/vi/tbWNXam34wI/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLBBxIuMdjc0wpTd5oPECiSCdNUOZA",
#             "https://lookaside.instagram.com/seo/google_widget/crawler/?media_id=3573051297643545119",
#             "https://media.npr.org/assets/img/2023/09/14/gettyimages-866151214_wide-76eb9e6773132611d45af8bb32fa0dd16088d2a9.jpg?s=1100&c=50&f=jpeg",
#             "https://i.ytimg.com/vi/MPb1VNt2EOo/maxresdefault.jpg",
#             "https://d28hgpri8am2if.cloudfront.net/book_images/onix/cvr9781668051351/what-this-comedian-said-will-shock-you-9781668051351_lg.jpg",
#             "https://cdnph.upi.com/svc/sv/upi/9221730999107/2024/1/78852f5b1f6cde4303514e743e208fdd/Bill-Maher-comedy-special-Is-Anyone-Else-Seeing-This-coming-in-January.jpg",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2024/05/BillMaherGettyImages-2074809903.jpg?w=1296&h=730&crop=1",
#             "https://m.media-amazon.com/images/S/pv-target-images/eddcbb81901ca6fc2fb76cee0dc3d67fc8eefb2b64cd18d719d60a26fdaee626.jpg",
#             "https://i.guim.co.uk/img/media/51c8821a9c1cfdbff33612063e293ba7432a591b/479_69_2515_1509/master/2515.jpg?width=465&dpr=1&s=none&crop=none",
#             "https://heroicmedia.org/wp-content/uploads/2024/04/bill-maher-scaled.webp",
#             "https://variety.com/wp-content/uploads/2025/03/maronmaher-split.jpg?w=1000&h=666&crop=1",
#             "https://9b16f79ca967fd0708d1-2713572fef44aa49ec323e813b06d2d9.ssl.cf2.rackcdn.com/1140x_a10-7_cTC/author-and-comedian-bill-maher-1720109364.jpg",
#             "https://www.portland5.com/sites/default/files/styles/event_square_large/public/events/2022/11/03/Static_Digital_438x400_BillMaher_2023_Regional_ArleneSchnitzerConcertHall_0402.jpg?itok=D04tgOjq",
#             "https://www.catholicleague.org/wp-content/uploads/2015/06/BillMaher.jpg",
#             "https://nypost.com/wp-content/uploads/sites/2/2025/03/bill-maher-100922151.jpg?quality=75&strip=all&w=1024",
#             "https://i.ticketweb.com/i/00/09/56/65/63/Original.jpg?v=1",
#             "https://static01.nyt.com/images/2019/10/06/magazine/06mag-talk-01/06mag-talk-01-videoSixteenByNine3000.jpg?year=2019&h=1687&w=3000&s=4e4bafdd643f7c96d411c7f1de4960f633e3351a53371769c473d426c4b14084&k=ZQJBKqZ0VN&tw=1",
#             "https://cloudfront-us-east-1.images.arcpublishing.com/advancelocal/MWFO34OGWNBPZHG34KPFYANLU4.jpg",
#             "https://resizing.flixster.com/-XZAfHZM39UwaGJIFWKAE8fS0ak=/v3/t/assets/1076_v9_bb.jpg",
#             "https://m.media-amazon.com/images/I/91jgsqDDw3L._AC_UF894,1000_QL80_.jpg",
#             "https://static01.nyt.com/images/2023/09/18/business/18strike-maher/18strike-maher-mediumSquareAt3X.jpg",
#             "https://i.scdn.co/image/ab67656300005f1f1ab33a96d6e81f73976168ef",
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Bill_Maher_by_David_Shankbone.jpg/1200px-Bill_Maher_by_David_Shankbone.jpg",
#             "https://i.ytimg.com/vi/V__wp4xvTjI/maxresdefault.jpg",
#             "https://deadline.com/wp-content/uploads/2022/01/Bill-Maher-2021-HBO.jpeg",
#             "https://www.rclco.com/wp-content/uploads/2023/11/Bill-Maher-Headshot-2023-Cropped.jpg",
#             "https://blog-admin.siriusxm.com/wp-content/uploads/2024/05/bill-maher-megyn-kelly3.jpg?w=885",
#             "https://www.usatoday.com/gcdn/-mm-/7268d9548b6d8c5c2f0290034b4c365680fd404b/c=0-107-1422-910/local/-/media/2017/06/10/USATODAY/USATODAY/636326590727276870-AP-TV-Bill-Maher.jpg?width=660&height=371&fit=crop&format=pjpg&auto=webp",
#             "https://people.com/thmb/4cdiKi_Ay3Kh8nAwgmsQDK8bAuA=/4000x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(234x0:236x2)/bill_maher470-c3978280f1794e2eb96d507667dcd344.jpg",
#             "https://mediaassets.cbre.com/-/media/project/cbre/shared-site/headshots/b/i/l/bill-maher-752x.jpg",
#             "https://www.sky.com/assets2/real-time-with-bill-maher-panel-ccbacc2b.jpg",
#             "https://thehill.com/wp-content/uploads/sites/2/2024/06/Joe-Biden-Bill-Maher-06.29.png?strip=1",
#             "https://images.nymag.com/news/features/maher120416_1_560.jpg",
#             "https://resizing.flixster.com/5IoKDUddaSoaIvNU0f3DgaZwci0=/ems.cHJkLWVtcy1hc3NldHMvbW92aWVzLzRlYzZmOGZiLWU5YjQtNDNlNC04N2RiLTUwYzllZTQ0ODYzZS5qcGc=",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2020/04/real_time_with_bill_maher.jpg?w=1296&h=730&crop=1",
#             "https://s.yimg.com/ny/api/res/1.2/YNLqG3jcObJi7U7o5sY6eQ--/YXBwaWQ9aGlnaGxhbmRlcjt3PTEyNDI7aD0xNDM4/https://media.zenfs.com/en/us_magazine_896/5f6f661f95aa4d2389c8e09c8441d720",
#             "https://i.ytimg.com/vi/V7R9h-SCX68/sddefault.jpg",
#             "https://cloudfront-us-east-2.images.arcpublishing.com/reuters/U3SVXGO5VJJVRM2EG2S627APPQ.jpg",
#             "https://images.cdn.prd.api.discomax.com/2025/01/22/a4e4a418-64f1-3bbc-889e-64fda09e7fcb.jpeg?f=jpg&q=75&w=1280",
#             "https://img.etimg.com/thumb/width-420,height-315,imgsize-3956,resizemode-75,msid-114404047/news/international/us/heres-why-bill-maher-desperately-wants-kamala-harris-to-win-the-us-elections-2024.jpg",
#             "https://www.thenation.com/wp-content/uploads/2015/03/bill_maher_ap_img.jpg",
#             "https://static01.nyt.com/images/2019/10/06/magazine/06mag-talk-03/06mag-talk-03-superJumbo-v2.jpg",
#             "https://deadline.com/wp-content/uploads/2021/05/Real-Time-with-Bill-Maher-2021.jpg?crop=54px%2C69px%2C1800px%2C1009px&resize=681%2C383",
#             "https://cdn.britannica.com/27/157827-050-F7F43E50/Bill-Maher-2011.jpg?w=400&h=300&c=crop",
#             "https://m.media-amazon.com/images/I/71o6EgaoYLL._AC_UF894,1000_QL80_.jpg",
#             "https://ew.com/thmb/98MJix5zx5yyE145IXgEJT0LaaM=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/real-time-with-bill-maher-3-2000-9f7cf5d3098d462785975c9c8380fd75.jpg",
#             "https://media.vanityfair.com/photos/63d7e820269be31f5c11ceda/16:9/w_2000,h_1125,c_limit/Bill-Maher.jpg",
#             "https://opinion-images.wsj.net/im-86981325/?size=1.5",
#             "https://www.politico.com/dims4/default/fbf2ff9/2147483647/legacy_thumbnail/1200x800%3E/quality/90/?url=http%3A%2F%2Fs3-origin-images.politico.com%2F2012%2F03%2F120316_bill_maher_facts_605_ap.jpg",
#             "https://static.independent.co.uk/2024/05/06/22/newFile-11.jpg",
#             "https://www.nydailynews.com/wp-content/uploads/migration/2009/10/26/6ZB6ZJIAZAWW7T62PFIOAWAJGQ.jpg?w=535",
#             "https://people.com/thmb/gP3f4pY-s7GWQQPNZEJsTEEhrP4=/4000x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(781x309:783x311)/Bill-Maher-011225-2-71f441edef3c498bb5657f9fb7814089.jpg",
#             "https://www.catholicleague.org/wp-content/uploads/2023/01/Maher-web-pic.png",
#             "https://www.onthisday.com/images/people/bill-maher.jpg?w=360",
#             "https://deadline.com/wp-content/uploads/2025/03/bill-maher-real-time-taylor-swift.jpg",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2024/05/maher-1024x576.jpg"
#           ]
#         },
#         {
#           "character": "Donald Trump",
#           "images": [
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/TrumpPortrait.jpg/1200px-TrumpPortrait.jpg",
#             "https://bidenwhitehouse.archives.gov/wp-content/uploads/2021/01/45_donald_trump_w-1250.jpg",
#             "https://cdn.britannica.com/88/190788-050-4F6F73F2/Donald-Trump-2016.jpg",
#             "https://hips.hearstapps.com/hmg-prod/images/gettyimages-2194420718-67d9b4e326598.jpg?crop=0.646xw:0.969xh;0.148xw,0&resize=640:*",
#             "https://www.aljazeera.com/wp-content/uploads/2025/03/AP25073719749030-1741984709.jpg?resize=770%2C513&quality=80",
#             "https://i.natgeofe.com/k/5e4ea67e-2219-4de4-9240-2992faef0cb6/trump-portrait.jpg",
#             "https://dims.apnews.com/dims4/default/8dba174/2147483647/strip/true/crop/5337x3563+0+0/resize/599x400!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F2b%2F65%2Fc1e9e86ec37b84b09b836148ea93%2Fba91ce472e28494693e16566e9a6fdd4",
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Donald_Trump_%288567815294%29_%28cropped%29.jpg/640px-Donald_Trump_%288567815294%29_%28cropped%29.jpg",
#             "https://i.abcnewsfe.com/a/030dec16-260c-4be0-8fe6-28b41c06bb36/donald-trump-9-gty-gmh-250313_1741885130014_hpMain.jpg",
#             "https://cdn.britannica.com/21/197021-050-D0AC3E38/portrait-Donald-Trump.jpg",
#             "https://c.files.bbci.co.uk/F332/production/_133585226_us24_flashcards_trump.png",
#             "https://people.com/thmb/1wrT7EDuoarBvk1MOgijDFM2AsE=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(999x0:1001x2)/barron-trump-donald-trump-inauguration-012025-3bf63b80cbe147388f5419c4a1f453f3.jpg",
#             "https://media-cldnry.s-nbcnews.com/image/upload/t_fit-560w,f_auto,q_auto:best/rockcms/2024-11/241108-donald-trump-al-0936-52ff7b.jpg",
#             "https://media.newyorker.com/photos/669324fd14c38da9ac7b7815/master/w_2560%2Cc_limit/Trump-shot-AP24195851874181.jpg",
#             "https://live-production.wcms.abc-cdn.net.au/48743062bb31e6708bdc6716e5a65628?impolicy=wcms_crop_resize&cropH=1316&cropW=2340&xPos=0&yPos=112&width=862&height=485",
#             "https://ru.usembassy.gov/wp-content/uploads/sites/138/AP25020618863748-1140x684.jpg",
#             "https://c8.alamy.com/comp/2N9GYRG/president-donald-trump-official-portrait-of-president-donald-j-trump-donald-john-trump-american-politician-media-personality-and-businessman-who-served-as-the-45th-president-of-the-united-states-from-2017-to-2021-2N9GYRG.jpg",
#             "https://static01.nyt.com/newsgraphics/2019/08/01/candidate-pages/3b31eab6a3fd70444f76f133924ae4317567b2b5/trump.jpg",
#             "https://www.aljazeera.com/wp-content/uploads/2025/03/AP25079748045798-1742504198.jpg?resize=770%2C513&quality=80",
#             "https://dobetter.esade.edu/sites/default/files/post/2024-10/victory-trump-EU-m.webp",
#             "https://m.media-amazon.com/images/I/81afCibN3PL.jpg",
#             "https://fortune.com/img-assets/wp-content/uploads/2016/04/tru05_a.jpg",
#             "https://i.abcnewsfe.com/a/6f82c457-4ae9-42bd-a9d6-dd430dabce19/donald-trump-10-ap-gmh-241106_1730906289206_hpMain_16x9.jpg?w=992",
#             "https://static01.nyt.com/images/2024/12/11/multimedia/11trump-news-time2-zpgt/11trump-news-time2-zpgt-mediumSquareAt3X.jpg",
#             "https://dims.apnews.com/dims4/default/c34ef7c/2147483647/strip/true/crop/4934x3294+0+0/resize/599x400!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fe7%2F74%2F2168cbc28168413d8179c6267766%2F992c8bdfe60643768471321b16c9ba98",
#             "https://m.media-amazon.com/images/I/61AobmicMsL.jpg",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2024/11/dt-winner-1024x585.jpg",
#             "https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=100044274887410",
#             "https://fortune.com/img-assets/wp-content/uploads/2025/02/GettyImages-2201327019-e1740324709905.jpg?w=1440&q=75",
#             "https://i.insider.com/6793e777e7e163809a2279ba?width=700",
#             "https://www.bu.edu/files/2025/01/trump-100-days-feat.jpg",
#             "https://snworksceo.imgix.net/dtc/df21c545-b4b8-4d85-ae96-507181573926.sized-1000x1000.jpg?w=1000",
#             "https://npr.brightspotcdn.com/dims3/default/strip/false/crop/4890x3260+0+0/resize/1100/quality/50/format/jpeg/?url=http%3A%2F%2Fnpr-brightspot.s3.amazonaws.com%2Feb%2F26%2F3424469c4d29b63b0b2683c3a4f2%2Fap24161743821924.jpg",
#             "https://time.com/ayit/images/7200212/cover.png",
#             "https://cdn.britannica.com/45/193845-050-A7905463/Donald-Trump-front-Tower-New-York-City-August-2008.jpg",
#             "https://media-cldnry.s-nbcnews.com/image/upload/t_fit-760w,f_auto,q_auto:best/rockcms/2025-02/250219-miami-beach-donald-trump-ac-935p-0b75a6.jpg",
#             "https://cdn.theatlantic.com/thumbor/YkmuUicCffTSqDzT6wL1DdyQxYo=/0x416:3712x2504/960x540/media/img/mt/2019/09/RTS2PCAV/original.jpg",
#             "https://deadline.com/wp-content/uploads/2019/05/donald-trump-the-apprentice.jpg?w=1000",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2024/11/GettyImages-2183216502-1024x683.jpg",
#             "https://media.vanityfair.com/photos/677e9d5ac1ae5d57ab0d79b9/master/w_2560%2Cc_limit/2192852996",
#             "https://cdn.abcotvs.com/dip/images/15515767_110524-cc-ap-trump-pointing-img.jpg",
#             "https://www.aljazeera.com/wp-content/uploads/2024/12/2024-12-07T175425Z_1928241406_RC2FKBARY8Z2_RTRMADP_3_USA-TRUMP-FRANCE-1733734296.jpg?resize=770%2C513&quality=80",
#             "https://s.france24.com/media/display/f40d9a2a-b89e-11ef-ae1b-005056a90284/w:1280/p:16x9/adf0caebdcbea6e5ca3304ecae700115f0d9086e.jpg",
#             "https://image.cnbcfm.com/api/v1/image/104466932-PE_Color.jpg?v=1737040322&w=800&h=600&ffmt=webp",
#             "https://m.media-amazon.com/images/I/81dG4y7IP7L._AC_UF1000,1000_QL80_.jpg",
#             "https://ca-times.brightspotcdn.com/dims4/default/303f45e/2147483647/strip/true/crop/4317x2883+0+0/resize/1200x801!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2Ff3%2Fa2%2F65eef0f049b8a8a14cd4f6f4bc6e%2Felection-2024-trump-95416.jpg",
#             "http://media.newyorker.com/photos/64b6d2167ac8745598305585/master/pass/Chotiner-Trump-Power.jpg",
#             "https://images.axios.com/B7ruxTmk9XUpH1NmSzPCSvUAXeg=/0x321:7008x4263/1920x1080/2024/04/30/1714481388445.jpg?w=3840",
#             "https://platform.vox.com/wp-content/uploads/sites/2/chorus/uploads/chorus_asset/file/7448639/cover.png?quality=90&strip=all&crop=0,7.1190476190476,100,85.761904761905",
#             "https://www.economist.com/cdn-cgi/image/width=1424,quality=80,format=auto/content-assets/images/20240706_USP503.jpg",
#             "https://si.usembassy.gov/wp-content/uploads/sites/113/2025/02/trump_profile_smithsonian-.jpeg",
#             "https://dims.apnews.com/dims4/default/e13ae1d/2147483647/strip/true/crop/3742x2495+0+0/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2Fb1%2Fd8%2F88d8271784c02ce9c32af6ae280f%2Fb815da18305c4c83904f8ecebacd7550",
#             "https://www.investopedia.com/thmb/kf9uXBLfpLnbZKZXkH6v0hmwZas=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/DonaldTrump-977a6dad59e34debb7a6ab3c92328510.jpg",
#             "https://cdn.cfr.org/sites/default/files/styles/slide_3_2/public/image/2024/06/Trump_final_1.jpg.webp",
#             "https://whyy.org/wp-content/uploads/2025/01/donald-trump-2025-01-08.jpg",
#             "https://news.berkeley.edu/wp-content/uploads/2025/03/APTOPIX_Trump_Zelenskyy_25059642193910-scaled.jpg",
#             "https://c.files.bbci.co.uk/4616/live/d710a6a0-d4b0-11ef-9fd6-0be88a764111.jpg",
#             "https://static.foxnews.com/foxnews.com/content/uploads/2024/11/donald-trump-campaign-msg.jpg",
#             "https://media.cnn.com/api/v1/images/stellar/prod/240119105200-donald-trump-0117.jpg?c=16x9&q=h_833,w_1480,c_fill",
#             "https://imageio.forbes.com/specials-images/imageserve/615db66e82a521d2906bb8e6/Donald-Trump/1960x0.jpg?format=jpg&width=960",
#             "https://media-cldnry.s-nbcnews.com/image/upload/t_fit-560w,f_auto,q_auto:best/rockcms/2025-02/250220-donald-trump-ac-1116p-d87512.jpg",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2025/02/2025-02-24T174606Z_1203290860_RC251DAHQ3D9_RTRMADP_3_USA-FRANCE-1024x682.jpg",
#             "https://whyy.org/wp-content/uploads/2025/03/trump-mccormick-ap-032325-768x512.jpg",
#             "https://www.politico.com/dims4/default/31b4beb/2147483647/strip/true/crop/3688x2459+0+0/resize/630x420!/quality/90/?url=https%3A%2F%2Fstatic.politico.com%2F3e%2F3f%2F8cc047fd4f6d900edaee705fcbd9%2Felection-2024-trump-43634.jpg",
#             "https://imagenes.elpais.com/resizer/v2/2PPMQOADCKQOGKLO3KFGNDPYMI.jpg?auth=e5741ec89d81b21a0c8c1cd67f4b68dc4f2c46482afbb5a07f007a1290ce6329&width=1960&height=1470&smart=true",
#             "https://i.abcnewsfe.com/a/25e5fdcf-6e28-41f5-a39d-969d1d8b34cd/inauguration-48-gty-gmh-250120_1737393470475_hpMain.jpg",
#             "https://assets.bwbx.io/images/users/iqjWHBFdfxIU/in5580yMORDI/v1/-1x-1.webp",
#             "https://s.france24.com/media/display/36ac61e6-00dd-11f0-8e1a-005056bf30b7/w:1280/p:16x9/063_2204326270.jpg",
#             "https://images.seattletimes.com/wp-content/uploads/2024/04/Y6TwKOO3.jpg?d=780x520",
#             "https://fortune.com/img-assets/wp-content/uploads/2025/01/GettyImages-2191378216_bf5b91-e1736185018887.jpg?w=1440&q=75",
#             "https://media.cnn.com/api/v1/images/stellar/prod/gettyimages-2174482169-copy.jpg?c=16x9&q=h_833,w_1480,c_fill",
#             "https://npr.brightspotcdn.com/dims3/default/strip/false/crop/1024x683+0+0/resize/1100/quality/50/format/jpeg/?url=http%3A%2F%2Fnpr-brightspot.s3.amazonaws.com%2F5d%2F65%2F6ab062cc4b2495b795e13556c4bf%2Fgettyimages-2194985045.jpg",
#             "https://static01.nyt.com/images/2024/11/05/multimedia/pol-trump-wins-hfo-new0101-qzgw/pol-trump-wins-hfo-new0101-qzgw-mediumSquareAt3X.jpg",
#             "https://media.newyorker.com/photos/671abac398e2f88da89a9d60/master/w_2560%2Cc_limit/Glasser-TrumpView.jpg",
#             "https://www.reuters.com/resizer/v2/65NBG2ZKS5IJJMJ4CTFYINSKRQ.jpg?auth=aa285593f752219d97af0b7693fd83892389623c6c7aeeabf471431f44d10a5f&width=7049&quality=80",
#             "https://media.nbcphiladelphia.com/2025/02/web-250209-donald-trump-getty.jpg?quality=85&strip=all&resize=1200%2C675",
#             "https://cdn.theatlantic.com/thumbor/5o08pfdzYI358Bx27p2MobrBek8=/0x0:3000x3750/648x810/media/img/2024/11/07/trump_normalize/original.jpg",
#             "https://dims.apnews.com/dims4/default/34eb7d8/2147483647/strip/true/crop/2143x1429+0+0/resize/599x399!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F86%2F6d%2F7729139dafde91820e1461acfce6%2Fce5cad3e4a514283b331e2aa89d7eb85",
#             "https://thumbnails.texastribune.org/k_5DHToac7q13Yb1mLFbOlGSNZY=/850x570/smart/filters:quality(75)/https://static.texastribune.org/media/files/210b3a66a97e03e7ca7acd2349408982/Trump%20and%20Melania%20REUTERS%20TT.jpg",
#             "https://lailluminator.com/wp-content/uploads/2024/02/TrumpSC-scaled-1-1536x1024.jpg",
#             "https://live-production.wcms.abc-cdn.net.au/09cc8f05ffa54eac464fb108453b4a58?impolicy=wcms_crop_resize&cropH=2813&cropW=5000&xPos=0&yPos=260&width=862&height=485",
#             "https://lookaside.instagram.com/seo/google_widget/crawler/?media_id=3539778042325295554",
#             "https://images.wsj.net/im-229976?size=1.5&width=640",
#             "https://www.hrw.org/sites/default/files/styles/embed_xxl/public/media_2025/01/202501usp_usa_donald_trump_executive_orders.jpg?itok=Zq-sUSVK",
#             "https://i.abcnewsfe.com/a/0f6aa83e-b2e3-4d9d-bae6-681d4a8585f3/apprentice-2-rt-er-250219_1740001709973_hpMain_9x16.jpg?w=992",
#             "https://deadline.com/wp-content/uploads/2025/02/madonna-donald-trump.jpg?w=681&h=383&crop=1",
#             "https://image.sciencenorway.no/2478477.webp?imageId=2478477&x=0.00&y=0.00&cropw=100.00&croph=100.00&width=960&height=644&format=jpg",
#             "https://www.bu.edu/files/2024/11/crop-AP24332089848604.jpg",
#             "https://ichef.bbci.co.uk/news/480/cpsprodpb/7304/live/22147890-ff9d-11ef-b0ab-6d56b02ced8f.jpg.webp",
#             "https://i.kinja-img.com/image/upload/c_fit,q_60,w_645/99b4fd6fc475aab45b3e25c97cc8eb3d.jpg",
#             "https://img.lemde.fr/2025/01/22/0/0/5435/3624/664/0/75/0/7ebe576_sirius-fs-upload-1-vdgaio8mmx2w-1737535305564-049554.jpg",
#             "https://images.theconversation.com/files/571418/original/file-20240125-27-a3g4ec.jpg?ixlib=rb-4.1.0&rect=0%2C0%2C2907%2C1926&q=20&auto=format&w=320&fit=clip&dpr=2&usm=12&cs=strip",
#             "https://www.americanprogress.org/wp-content/uploads/sites/2/2025/02/GettyImages-2201118913cropped.jpg",
#             "https://platform.vox.com/wp-content/uploads/sites/2/2024/07/GettyImages-2161922012.jpg?quality=90&strip=all&crop=0.021331058020472%2C0%2C99.957337883959%2C100&w=2400",
#             "https://www.usnews.com/dims4/USNEWS/3516ede/2147483647/thumbnail/970x647/quality/85/?url=https%3A%2F%2Fwww.usnews.com%2Fcmsmedia%2Fb4%2Ff9%2F0835906340ffbb3196a84f58ae60%2F170921-donaldtrump10things-editorial.jpg",
#             "https://cdn.britannica.com/31/149831-050-83A0E45B/Donald-J-Trump-2010.jpg",
#             "https://www.thenation.com/wp-content/uploads/2024/07/Donald-Trump-Department-of-Education.jpg",
#             "https://www.citizensforethics.org/wp-content/uploads/2021/02/46451130962_d151b497d6_b-1200x675.jpg",
#             "https://www.thoughtco.com/thmb/OIQxajVv1AMskdgr21fgM-qpCOo=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-1163026796-ef8fdfa1a68a471d8bda5aeab73c98c0.jpg",
#             "https://people.com/thmb/-Z24JV9E4vfl2olDngYIcN8JLdc=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(999x0:1001x2)/donald-trump-melania-121224-3da0aec3465f4588b25d7217926c431b.jpg"
#           ]
#         }
#       ],
#       "section_videos": [
#         {
#           "Character": "Bill Maher",
#           "videos": [
#             {
#               "title": "Main Video",
#               "url": "https://youtu.be/vRz47H4iFv4"
#             },
#             {
#               "title": "Economic Ozempic | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=c9txhe-5EJA"
#             },
#             {
#               "title": "New Rule: The Next Democratic Star | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=6OyDdTJDqTQ"
#             },
#             {
#               "title": "Jim Carrey | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=ZkFUU-xJM6I"
#             },
#             {
#               "title": "Bill Maher, Kid Rock, and Trump at Dinner – Imagine Being a Fly on That Wall!",
#               "url": "https://www.youtube.com/watch?v=qcJZpNVqGII"
#             },
#             {
#               "title": "Bill Maher Left SPEECHLESS After Confronting Guest On Regretting Voting For Trump BACK FIRES!",
#               "url": "https://www.youtube.com/watch?v=jj7fqHviIWg"
#             },
#             {
#               "title": "Bill Maher DEFENDS Trump’s Savage Response to Reporter",
#               "url": "https://www.youtube.com/watch?v=0gq1iMfGhis"
#             },
#             {
#               "title": "Was Bill Maher right about Democrats? | On Balance",
#               "url": "https://www.youtube.com/watch?v=zGEtHd52j9Q"
#             },
#             {
#               "title": "Chris Cuomo BLASTS Don Lemon Over Calling Black MAGA Supporters Not ‘Rational’ On Bill Maher",
#               "url": "https://www.youtube.com/watch?v=gPz8rBEol_A"
#             },
#             {
#               "title": "New Rule: Political Firestorm | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=C5S8rhNCBnc"
#             },
#             {
#               "title": "Bill Maher RIPS Dept. Of Education. Should Agency Exist? Rising Interview",
#               "url": "https://www.youtube.com/watch?v=8zWYFQx5Ai0"
#             },
#             {
#               "title": "Bill Maher Asks 'What Does The Department of Education Do?'",
#               "url": "https://www.youtube.com/watch?v=MJieLa6zUlg"
#             },
#             {
#               "title": "New Rule: Don't Go to College | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=fMF0bser1aM"
#             },
#             {
#               "title": "New Rule: Climate Shame | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=63KXfwC9BdU"
#             },
#             {
#               "title": "Real Time with Bill Maher: Meet the Climate Change Deniers (HBO)",
#               "url": "https://www.youtube.com/watch?v=Mma_PJqYe24"
#             },
#             {
#               "title": "Bill Maher on Climate Change Denial",
#               "url": "https://www.youtube.com/watch?v=ReNUJ3c1k68"
#             }
#           ]
#         },
#         {
#           "Character": "Donald Trump",
#           "videos": [
#             {
#               "title": "BREAKING NEWS: JD Vance Issues Emphatic Defense Of Trump Tariffs In Speech On Economic Dynamism",
#               "url": "https://www.youtube.com/watch?v=o3MLLCYii4Q"
#             },
#             {
#               "title": "Trump makes rare presidential appearance at Justice Department",
#               "url": "https://www.youtube.com/watch?v=SwDZ83uBsK4"
#             },
#             {
#               "title": "'UNRELENTING': Trump official sends 'clear' message to Iran-backed Houthis",
#               "url": "https://www.youtube.com/watch?v=SE-H5mG4Qso"
#             },
#             {
#               "title": "Bill Maher Left SPEECHLESS After Getting EDUCATED ON Trump’s TARIFFS!",
#               "url": "https://www.youtube.com/watch?v=KQfrHvDpyfc"
#             },
#             {
#               "title": "BREAKING`: Bill Maher Just CONFRONTED Donald Trump On LIVE TV!",
#               "url": "https://www.youtube.com/watch?v=jSc6epDwm-g"
#             },
#             {
#               "title": "Bill Maher NUKES Trump in HEATED DEBATE with MAGA Guest!",
#               "url": "https://www.youtube.com/watch?v=-snk3FsJ7d4"
#             },
#             {
#               "title": "Trump to Discuss Russia-Ukraine Ceasefire with Putin Over Phone Call | Firstpost America | N18G",
#               "url": "https://www.youtube.com/watch?v=KGaQ2HFb6tI"
#             },
#             {
#               "title": "President Trump, Putin discuss ceasefire over phone",
#               "url": "https://www.youtube.com/watch?v=_kN-NwBxJTk"
#             },
#             {
#               "title": "Donald Trump & Vladimir Putin Expected To Discuss Ending The War In Ukraine | N18G",
#               "url": "https://www.youtube.com/watch?v=MunZJ-bskCc"
#             },
#             {
#               "title": "Alert! Trump threatens political ‘enemies’ in chilling DOJ speech",
#               "url": "https://www.youtube.com/watch?v=0fFezDZ1jaM"
#             },
#             {
#               "title": "Judge’s order ignored, voters sour on inflation: Trump’s First 100 Days - Day 57 | MSNBC Highlights",
#               "url": "https://www.youtube.com/watch?v=5YFyC4iDxJw"
#             },
#             {
#               "title": "Reporter asks Trump if he defied judge's court order",
#               "url": "https://www.youtube.com/watch?v=amUgI1dJoaw"
#             },
#             {
#               "title": "38 Republicans sink Trump-backed spending bill",
#               "url": "https://www.youtube.com/watch?v=vaR8MQz-VBs"
#             },
#             {
#               "title": "Why Trump's budget, could devastate Medicaid",
#               "url": "https://www.youtube.com/watch?v=Se2HAEs2W2Y"
#             },
#             {
#               "title": "Democrats Harshly Criticize Trump's 2018 Budget",
#               "url": "https://www.youtube.com/watch?v=ME668mZ_3ME"
#             }
#           ]
#         }
#       ]
#     },
#     "body": {
#       "section": "body",
#       "section_script": "Quite surprising, isn't it? I get it, California is always at the forefront, pioneering trends that others might adopt, but here's Maher wondering, \"Is this the traditional stance of the Democratic party on immigration?\" Immigration has long been a topic both parties engaged with. Consider this, while we all support spreading the American Dream, shouldn't it ideally be reserved for Americans? Maher is skeptically lifting an eyebrow, seemingly questioning whether the guidelines have perhaps shifted too significantly. For Bill, the pieces just aren't adding up. He’s not against aiding others, yet he's delving deep, addressing the hard-hitting questions. Gender Discussions Next up, the ongoing debate surrounding gender inclusion in sports. President Trump has introduced an executive order to prevent boys from entering girls’ sports competitions. The response was mixed, some applauded, others frowned upon it. But Maher? He's jumping into the fray, blending his trademark humor with incisive commentary. Instead of taking a stand, you know what he does..? [CLIP: President Trump this week just did an executive order saying that boys can no longer play in girls' sports. That's it. That's it. Ladies, isn't it ironic that the biggest protector of women turned out to be the pussy grabber? Huh? Maybe the guy who grabs it knows how to protect it the best. Think about that. Think about that. ] Maher is really targeting the paradox in the situation. It's ironic how Trump is viewed as a defender of women, considering his notorious history with scandals. Provoking Thoughts What I find fascinating about Maher is his approach. It goes beyond just irritation or humor. He genuinely stimulates reflection on cultural standards. Meanwhile, he includes everyone in the humor, ensuring they stick around for the inevitable tweeting frenzy that follows. And guess what? There’s even more from Maher. His latest critiques of progressives form quite the spectacle. This time around, he's exposing what he perceives as brainwashing among today's youth. He’s linking their backing of Hamas, a notably controversial opinion, with their anti-American views. You might not believe it, but Bill is advocating for a resurgence of patriotic values in education. He’s endorsing an educational system that fosters love for America, rather than disdain. In one of his classic criticisms, Maher has even spotlighted Senate Majority Leader Chuck Schumer’s concerns over anti-Semitism, an issue that is extremely relevant today. Maher is echoing Schumer’s concerns about dangerous chants and their implications regarding erasure. Watching this develop is quite alarming, to be honest. Maher is prominently supporting, reinforcing the message that Schumer has established. [CLIP: Under his administration, his next one, schools will teach students to love their country, not to hate their country, like they're taught now. I got to say, this one doesn't bother me so much, because I think that, I mean, this is what I see when I see kids demonstrating these progressives, demonstrating for Hamas, the most illiberal people in the world, that, oh, good, we're going to give America its comeuppance, asshole America. This is, they kind of have been indoctrinated this way. And Chuck Schumer made a speech this week. He said anti-Semitism is a five-alarm fire that must be extinguished. This is the highest-ranking Jewish politician we've ever had in this country. He talked about when Jewish people hear chants like, from the river to the sea, you understand that means wipe us out by any means necessary. Vulnerabilities in the party, James. Democratic Contradictions ] In the midst of all the chaos, there's a palpable sense of exasperation within the Democratic party. Isn't it astonishing how Democrats often entangle themselves in their own contradictions? Take Maher’s razor-sharp commentary, for example—it’s utterly mind-blowing. Does this sound familiar? Democrats seem to find themselves navigating challenges in a convoluted and sometimes outright illogical manner. And there’s Maher, schooling everyone on how to tackle issues directly! It's clear why his straightforward and candid insights are gaining attention. So, I have to wonder... what are your thoughts? Are you stunned by Maher’s bold style, or do you see the cleverness in his remarks? Leave your thoughts below and share your views on Maher's characterization of education as a \"black hole\" or his probing questions about who really benefits from the American Dream. Eagerly awaiting your input!",
#       "imageData": [
#         {
#           "characterName": "President Trump",
#           "images": [
#             "https://kansasreflector.com/wp-content/uploads/2024/07/GettyImages-2162619255-2048x1365-1-1536x1024.jpg",
#             "https://whyy.org/wp-content/uploads/2024/11/donald-trump-election-presidential-race-AP-110624.jpg",
#             "https://www.thinkglobalhealth.org/sites/default/files/styles/max_325x325/public/2025-01/2025-01-27T165747Z_1114190312_RC21ECAOF4QD_RTRMADP_3_USA-TRUMP-LGBT-LAWSUIT.JPG?itok=egu0Xugl",
#             "https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iCy6sk0oBAIE/v1/-1x-1.webp",
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/TrumpPortrait.jpg/800px-TrumpPortrait.jpg",
#             "https://bidenwhitehouse.archives.gov/wp-content/uploads/2021/01/45_donald_trump.jpg",
#             "https://i.insider.com/6797aeb3196626c409851836?width=1200&format=jpeg",
#             "https://www.usda.gov/sites/default/files/styles/two_column_header_image_1320x888/public/sec-rollins-nat-ag-day-pres-trump.jpg?h=c6980913&itok=62Vk6pFG",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2024/11/dt-winner-1024x585.jpg",
#             "https://d.newsweek.com/en/full/2613100/elon-musk-donald-trump.jpg?w=1200&f=f32e1a1e27eb66aad8e734e63e217995",
#             "https://cdn.britannica.com/21/197021-050-D0AC3E38/portrait-Donald-Trump.jpg",
#             "https://nypost.com/wp-content/uploads/sites/2/2025/03/united-states-president-donald-trump-100413395.jpg?quality=75&strip=all&w=744",
#             "https://s.abcnews.com/images/US/donald-trump-10-ap-gmh-241106_1730906289206_hpMain_16x9_1600.jpg",
#             "https://i.natgeofe.com/k/5e4ea67e-2219-4de4-9240-2992faef0cb6/trump-portrait.jpg",
#             "https://media.newyorker.com/photos/67902dac62c7d191667ff4ad/master/w_2560%2Cc_limit/Gersen-TrumpSignings-Mills.jpg",
#             "https://static01.nyt.com/newsgraphics/2019/08/01/candidate-pages/3b31eab6a3fd70444f76f133924ae4317567b2b5/trump.jpg",
#             "https://idahocapitalsun.com/wp-content/uploads/2024/11/trumpnov13-scaled.jpg",
#             "https://c.files.bbci.co.uk/4616/live/d710a6a0-d4b0-11ef-9fd6-0be88a764111.jpg",
#             "https://npr.brightspotcdn.com/dims3/default/strip/false/crop/1024x732+0+0/resize/1100/quality/50/format/jpeg/?url=http%3A%2F%2Fnpr-brightspot.s3.amazonaws.com%2F00%2Fb9%2F4c96b78f469a868605baad1a185c%2Fgettyimages-2203857531.jpg",
#             "https://ru.usembassy.gov/wp-content/uploads/sites/138/AP25020618863748-scaled.jpg",
#             "https://www.reuters.com/resizer/v2/UUYZET5LUJINVGAODHS5PI2XNQ.jpg?auth=9cd79b0ffff001238d732400bd2701cb088bd0df4c05413ea801140e30d9d01c&width=5198&quality=80",
#             "https://media.cnn.com/api/v1/images/stellar/prod/gettyimages-2205875388.jpg?q=w_1110,c_fill",
#             "https://static.scientificamerican.com/dam/m/530f6a83ec06a4d4/original/Trump_signs_executive_orders.jpg?m=1737561002.57&crop=16%3A9%2Csmart&w=1920",
#             "https://media.vanityfair.com/photos/6371861f57377e33b6f57199/master/w_2560%2Cc_limit/1431869535",
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Donald_Trump_%288567815294%29_%28cropped%29.jpg/1200px-Donald_Trump_%288567815294%29_%28cropped%29.jpg",
#             "https://www.reuters.com/resizer/v2/KSPOO6STX5LFVNXDWHOA7WXAP4.jpg?auth=24b60ea0f5a37148dd3742ab6080f7756ba2000a574025ae98073eea7ebd5015&width=1080&quality=80",
#             "https://people.com/thmb/Ur9UpqkezoBaDUb7y-xy6fGJA8Q=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(753x318:755x320)/donald-trump-shut-up-about-egg-prices-031125-5fedea9793b24175a12ddc5db2984d82.jpg",
#             "https://si.usembassy.gov/wp-content/uploads/sites/113/2025/02/trump_profile_smithsonian-.jpeg",
#             "https://ca-times.brightspotcdn.com/dims4/default/28245b3/2147483647/strip/true/crop/3000x2000+0+0/resize/1200x800!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2F71%2Ff0%2F6aba34814415ad71445b4e70b94f%2Feeuu-elecciones-opiniones-de-votantes-00030.jpg",
#             "https://i.abcnewsfe.com/a/0f6aa83e-b2e3-4d9d-bae6-681d4a8585f3/apprentice-2-rt-er-250219_1740001709973_hpMain_9x16.jpg?w=992",
#             "https://defensescoop.com/wp-content/uploads/sites/8/2024/12/Trump_d7ed75.jpg?w=1013",
#             "https://www.pewresearch.org/wp-content/uploads/sites/20/2025/03/SR_25.03.03_sotu_feature.jpg?w=640",
#             "https://npr.brightspotcdn.com/dims3/default/strip/false/crop/5236x3491+0+0/resize/1100/quality/50/format/jpeg/?url=http%3A%2F%2Fnpr-brightspot.s3.amazonaws.com%2F75%2F46%2F50d8174944668a6411b3b1fb4003%2Fap25064161390673.jpg",
#             "https://snworksceo.imgix.net/dtc/df21c545-b4b8-4d85-ae96-507181573926.sized-1000x1000.jpg?w=1000&ar=16%3B9&fit=crop&crop=faces&facepad=3&auto=format",
#             "https://cdn.theatlantic.com/thumbor/fhigyO0dP3BlYDmx2H4g89kkA5o=/0x281:5574x3416/960x540/media/img/mt/2020/03/RTS37HPB-1/original.jpg",
#             "https://images.theconversation.com/files/643798/original/file-20250121-15-mhxlbo.jpg?ixlib=rb-4.1.0&rect=0%2C86%2C7185%2C4707&q=20&auto=format&w=320&fit=clip&dpr=2&usm=12&cs=strip",
#             "https://lailluminator.com/wp-content/uploads/2024/02/TrumpSC-scaled-1-1536x1024.jpg",
#             "https://dims.apnews.com/dims4/default/adb221b/2147483647/strip/true/crop/5716x3815+0+0/resize/599x400!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F20%2F8c%2F0a9ef8a741a8bea73ae6619c58d7%2Fap25042756828063.jpg",
#             "https://platform.vox.com/wp-content/uploads/sites/2/2025/03/GettyImages-2203829700.jpg?quality=90&strip=all&crop=0%2C0%2C100%2C100&w=2400",
#             "https://d.newsweek.com/en/full/2572575/trump.jpg?w=1200&f=a8ce696455921d925bfaca5d9118b82d",
#             "https://www.bu.edu/files/2024/11/trump-research-feat.jpg",
#             "https://ca-times.brightspotcdn.com/dims4/default/abeeca8/2147483647/strip/true/crop/6000x4000+0+0/resize/1200x800!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2Fca%2F81%2F059f33084d28b73bcd0d100cef87%2Ftrump-61047.jpg",
#             "https://imageio.forbes.com/specials-images/imageserve/66e9ea2f67d3fefc9fbe1239/4x5-Donald-Trump-by-Spencer-Platt-Getty-Images/0x0.jpg?format=jpg&height=2401&width=1920",
#             "https://platform.vox.com/wp-content/uploads/sites/2/chorus/uploads/chorus_asset/file/25274440/1965959342.jpg?quality=90&strip=all&crop=0%2C0%2C100%2C100&w=2400",
#             "https://fortune.com/img-assets/wp-content/uploads/2025/02/GettyImages-2201780017-e1740583416448.jpg?w=1440&q=75",
#             "https://michiganadvance.com/wp-content/uploads/2025/01/trumpjan20-1536x1024.jpg",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2025/01/2025-01-29T172751Z_1854301353_RC2VFCA9S0DW_RTRMADP_3_BRITAIN-TRUMP-COURT-1024x683.jpg",
#             "https://www.aljazeera.com/wp-content/uploads/2024/11/AP24309574257630-1730736613.jpg?resize=1920%2C1080",
#             "https://media-cldnry.s-nbcnews.com/image/upload/t_fit-760w,f_auto,q_auto:best/rockcms/2025-03/250311-donald-trump-tesla-ew-444p-6d3252.jpg",
#             "https://cdn.cfr.org/sites/default/files/styles/full_width_xl/public/image/2025/01/Trump%20Rubio.webp",
#             "https://s.france24.com/media/display/1c3a99ce-6556-11ed-9bd5-005056a90284/w:1280/p:16x9/2022-11-16T022244Z_543999449_RC2QMX95YN5H_RTRMADP_3_USA-ELECTION-TRUMP.JPG",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2025/03/1200/675/trump-speech.jpg?ve=1&tl=1",
#             "https://whyy.org/wp-content/uploads/2025/02/trump-npr-021225-768x512.jpeg",
#             "https://www.science.org/do/10.1126/science.zxvum0n/full/_20250121_nid_trump_who-1737648919077.jpg",
#             "https://www.thoughtco.com/thmb/OIQxajVv1AMskdgr21fgM-qpCOo=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-1163026796-ef8fdfa1a68a471d8bda5aeab73c98c0.jpg",
#             "https://media.cnn.com/api/v1/images/stellar/prod/gettyimages-2194383933.jpg?c=16x9&q=h_833,w_1480,c_fill",
#             "https://www.citizensforethics.org/wp-content/uploads/2020/09/50349050837_06318c5780_b-1200x675.jpg",
#             "https://www.brookings.edu/wp-content/uploads/2025/03/2025-03-05T141645Z_1497928456_MT1USATODAY25585055_RTRMADP_3_MAR-4-2025-WASHINGTON-DC-USA-PRESIDENT-DONALD-TRUMP-1.jpg?quality=75&w=1500",
#             "https://www.economist.com/cdn-cgi/image/width=1424,quality=80,format=auto/content-assets/images/20240608_USP002.jpg",
#             "https://c.files.bbci.co.uk/C51D/production/_133516405_updatedpromoimage.png",
#             "https://cdn.theatlantic.com/thumbor/5o08pfdzYI358Bx27p2MobrBek8=/0x0:3000x3750/648x810/media/img/2024/11/07/trump_normalize/original.jpg",
#             "https://hips.hearstapps.com/hmg-prod/images/gettyimages-2194420718-67d9b4e326598.jpg?crop=0.646xw:0.969xh;0.148xw,0&resize=640:*",
#             "https://media-cldnry.s-nbcnews.com/image/upload/t_fit-1500w,f_auto,q_auto:best/rockcms/2023-09/230921-donald-trump-jm-1109-57b054.jpg",
#             "https://www.citizensforethics.org/wp-content/uploads/2021/02/46451130962_d151b497d6_b-1200x675.jpg",
#             "https://physicsworld.com/wp-content/uploads/2024/12/Trump-re-election-6-Nov-2024-small.jpg",
#             "https://cdn.abcotvs.com/dip/images/14890421_053024-cc-trump-ap-img.jpg",
#             "https://media.vanityfair.com/photos/672a497a98a8cfd30ca61fde/master/w_2560%2Cc_limit/fv1124-trump.jpg",
#             "https://assets.medpagetoday.net/media/images/111xxx/111859.jpg?width=0.8",
#             "https://people.com/thmb/IUi55iAYATKHqTaxTKnY8tPWCc8=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(749x191:751x193)/Donald-Trump-Grover-Cleveland-Tout-102824-d5e8b58f46e442879578752bcd467325.jpg",
#             "https://m.economictimes.com/thumb/msid-115025097,width-1600,height-900,resizemode-4,imgsize-45320/donald-trump-is-usa-president-elect-what-does-it-mean-for-the-world.jpg",
#             "https://npr.brightspotcdn.com/dims3/default/strip/false/crop/5816x3877+0+0/resize/1100/quality/50/format/jpeg/?url=http%3A%2F%2Fnpr-brightspot.s3.amazonaws.com%2F33%2Fef%2Fcad7a8bf4e899704c0c379f22653%2Fgettyimages-2203683508.jpg",
#             "https://media.defense.gov/2019/Aug/30/2002177620/1920/1080/0/190829-D-BN624-0255.JPG",
#             "https://www.aljazeera.com/wp-content/uploads/2024/12/2024-12-07T175425Z_1928241406_RC2FKBARY8Z2_RTRMADP_3_USA-TRUMP-FRANCE-1733734296.jpg?resize=770%2C513&quality=80",
#             "http://media.newyorker.com/photos/5e6a47215a2694000896ab00/master/pass/Glasser-TrumpCoronavirusSpeech.jpg",
#             "https://bsmedia.business-standard.com/_media/bs/img/article/2024-11/15/full/1731662391-1183.jpg?im=FeatureCrop,size=(826,465)",
#             "https://images.wsj.net/im-82923306/?width=1280&height=1280",
#             "https://cdn.abcotvs.com/dip/images/15821493_trump-orders-img.png",
#             "https://images.newrepublic.com/43817d344cc9aa0f15fd78e6c8349809bb904b6f.jpeg?auto=format&fit=crop&crop=faces&q=65&w=768&ar=3%3A2&ixlib=react-9.10.0",
#             "https://www.brookings.edu/wp-content/uploads/2024/10/2024-09-11T054926Z_2092926145_MT1USATODAY24204474_RTRMADP_3_SEP-10-2024-PHILADELPHIA-PA-USA-FORMER-PRESIDENT-DONALD1.jpg?quality=75&w=1500",
#             "https://vn.usembassy.gov/wp-content/uploads/sites/124/2025/01/President-Trump-priorities.jpg",
#             "https://static.euronews.com/articles/stories/08/76/78/88/1200x675_cmsv2_4a4d8afe-2d30-5d5c-aa84-13546c1a64d6-8767888.jpg",
#             "https://fortune.com/img-assets/wp-content/uploads/2025/01/GettyImages-2194398603.jpg?w=1440&q=75",
#             "https://img-cdn.inc.com/image/upload/f_webp,q_auto,c_fit/vip/2024/10/donald-trump-wins-election-inc.jpg",
#             "https://static01.nyt.com/images/2024/11/05/multimedia/pol-trump-wins-hfo-new0101-qzgw/pol-trump-wins-hfo-new0101-qzgw-mediumSquareAt3X.jpg",
#             "https://www.americanprogress.org/wp-content/uploads/sites/2/2017/04/AP_17115159898594-PREPPED.jpg?w=1040",
#             "https://dobetter.esade.edu/sites/default/files/post/2024-10/victory-trump-EU-w.webp",
#             "https://s.france24.com/media/display/8eb0b174-9c4e-11ef-8886-005056bfb2b6/w:1280/p:16x9/main-nebuleuse-trump.jpg",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2025/02/2025-02-24T174606Z_1203290860_RC251DAHQ3D9_RTRMADP_3_USA-FRANCE-1024x682.jpg",
#             "https://www.whitehouse.gov/wp-content/themes/whitehouse/assets/img/whitehouse-47-donald-j-trump-cover.png",
#             "https://variety.com/wp-content/uploads/2024/12/GettyImages-2188144000.jpg?w=1000&h=667&crop=1"
#           ]
#         },
#         {
#           "characterName": "Bill Maher",
#           "images": [
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/BillMaherSept10.jpg/1200px-BillMaherSept10.jpg",
#             "https://static.hbo.com/2024-12/real-time-with-bill-maher-s23-ka-1920.jpg",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2024/05/1200/675/BillMaher53124.jpg?ve=1&tl=1",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2025/03/1200/675/mahermuskchainsaw2.jpg?ve=1&tl=1",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2022/05/1200/675/d849cd06-Maher.jpg?ve=1&tl=1",
#             "https://thecomedystore.com/wp-content/uploads/2023/06/e5984e4c255ee9ceee6f7c2fde95508a.jpg",
#             "https://edge.dailyvoice.com/upload/FotoJet_HnzrXOO.jpg?width=720",
#             "https://static.hbo.com/2022-04/bill-maher-adulting-ka-1920.jpg?w=1200",
#             "https://www.nj.com/resizer/v2/N72EFUO2I5A5PKGQ4WNP55CRJU.jpg?auth=4c2b5eb7ee833d3e224fb747b26cc422ee5cf8f00af6479f1cfae7380dcc2b5f&width=500&quality=90",
#             "https://cdn.choosechicago.com/uploads/2024/10/Bill-Maher-BIC-1280x820-1.jpg",
#             "https://media.gq.com/photos/5c7571166a6e6372b18b6e67/16:9/w_2560%2Cc_limit/Bill-Maher-Retire-Bitch-GQ-2019-022619.jpg",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2021/09/bill-maher-andrew-yang-H-2021.jpg",
#             "https://latenighter.com/wp-content/uploads/2024/07/TSDSARA_EC006.jpg",
#             "https://ffrf.org/wp-content/uploads/1980/01/90c519c0607749e3c2d669154a64428f.jpg",
#             "https://www.thekeyexecutives.com/wp-content/uploads/2021/08/Bill-Maher.jpg",
#             "https://m.media-amazon.com/images/M/MV5BNTQ4MzgwMTcxMl5BMl5BanBnXkFtZTcwNjAyNzQzMg@@._V1_FMjpg_UX1000_.jpg",
#             "https://variety.com/wp-content/uploads/2017/01/real-time-with-bill-maher.jpg?w=1000&h=562&crop=1",
#             "https://people.com/thmb/CSefpKe40XHzdl7qwhpoQyWtqB4=/4000x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(535x0:537x2)/bill-maher--Noor-Alfallah-1-100924-a7a0c0ab3b4f46dd8606832a1990eb63.jpg",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2024/04/1200/675/MaherTrump.jpg?ve=1&tl=1",
#             "https://media.zenfs.com/en/parade_250/9d6e4237175a3d23199b5a3c8499c057",
#             "https://cbsaustin.com/resources/media2/16x9/full/1015/center/80/09186cb0-09e0-4fb3-9b25-5f9347afec0e-large16x9_AP21134082736586.jpg",
#             "https://external-preview.redd.it/its-an-honor-bill-maher-explains-why-hes-going-to-meet-soon-v0-s3QJw9zggIIDHmroTqYVCLKppOMeZGTkVvIqoChe7cs.jpg?width=640&crop=smart&auto=webp&s=1f860ffe64573313f84cdeb5e8669d3f11ef7860",
#             "https://m.media-amazon.com/images/S/pv-target-images/c6b83ca34a7c8ddd424a0ccb97d2cfc1bec750906d05f50ac4666dd7a7daf3d2._SX1080_FMjpg_.jpg",
#             "https://static.hbo.com/content/dam/hbodata/series/real-time-with-bill-maher/episodes/15/800603_RTWBM_S15_L0_3000x1688.jpg",
#             "https://deadline.com/wp-content/uploads/2025/03/Bill-Maher-PA-Gov.-Josh-Shapiro.jpg",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2024/03/Bill-Maher-getty-H-2024.jpg?w=1296&h=730&crop=1",
#             "https://kleinhansbuffalo.org/wp-content/uploads/sites/2/2022/03/Bill-Maher-credit-David-Becker-WireImage-11.25.13-scaled.jpg",
#             "https://variety.com/wp-content/uploads/2024/11/GettyImages-2153895970-e1730996650378.jpg?w=1000&h=667&crop=1",
#             "https://dw-wp-production.imgix.net/2025/03/GettyImages-2202961976-scaled.jpg?ar=16:9&fit=crop&crop=faces&fm=webp",
#             "https://people.com/thmb/QgC19bpHBlHGR-CodUM6tV1T5hQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(999x0:1001x2)/bill-maher-1d579295ab1e4fa7b361b5dca2627685.jpg",
#             "https://i0.wp.com/www.nydailynews.com/wp-content/uploads/2024/10/241022_maher2.jpg?fit=620%2C9999px&ssl=1",
#             "https://m.media-amazon.com/images/M/MV5BMTUxNTY4NzIwMl5BMl5BanBnXkFtZTgwNDIxMTQ5NjE@._V1_.jpg",
#             "https://d.newsweek.com/en/full/2603083/bill-maher.jpg?w=1600&h=1200&q=88&f=a39cb4f94b906da2034e9d784c6d2941",
#             "https://i.ytimg.com/vi/MPb1VNt2EOo/maxresdefault.jpg",
#             "https://media.npr.org/assets/img/2023/09/14/gettyimages-866151214_wide-76eb9e6773132611d45af8bb32fa0dd16088d2a9.jpg?s=1100&c=50&f=jpeg",
#             "https://heroicmedia.org/wp-content/uploads/2024/04/bill-maher-scaled.webp",
#             "https://pbs.twimg.com/ext_tw_video_thumb/1903280147840765952/pu/img/yWyF4WccypovuG3k.jpg:large",
#             "https://cdnph.upi.com/svc/sv/upi/9221730999107/2024/1/78852f5b1f6cde4303514e743e208fdd/Bill-Maher-comedy-special-Is-Anyone-Else-Seeing-This-coming-in-January.jpg",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2024/05/BillMaherGettyImages-2074809903.jpg?w=1296&h=730&crop=1",
#             "https://www.advocate.com/media-library/trans-comic-on-bill-maher-grow-up-and-go-away.jpg?id=32678112&width=1245&height=700&coordinates=0%2C47%2C0%2C96",
#             "https://www.pabsttheatergroup.com/assets/img/2024.07.14-R-Bill-Maher500x500-Web-Thumbnail-d8b5d09e68.png",
#             "https://i.guim.co.uk/img/media/51c8821a9c1cfdbff33612063e293ba7432a591b/479_69_2515_1509/master/2515.jpg?width=465&dpr=1&s=none&crop=none",
#             "https://variety.com/wp-content/uploads/2025/03/maronmaher-split.jpg?w=1000&h=666&crop=1",
#             "https://9b16f79ca967fd0708d1-2713572fef44aa49ec323e813b06d2d9.ssl.cf2.rackcdn.com/1140x_a10-7_cTC/author-and-comedian-bill-maher-1720109364.jpg",
#             "https://nypost.com/wp-content/uploads/sites/2/2025/03/bill-maher-100922151.jpg?quality=75&strip=all&w=1024",
#             "https://www.catholicleague.org/wp-content/uploads/2015/06/BillMaher.jpg",
#             "https://www.thenation.com/wp-content/uploads/2015/03/bill_maher_ap_img.jpg",
#             "https://resizing.flixster.com/-XZAfHZM39UwaGJIFWKAE8fS0ak=/v3/t/assets/1076_v9_bb.jpg",
#             "https://cloudfront-us-east-1.images.arcpublishing.com/advancelocal/MWFO34OGWNBPZHG34KPFYANLU4.jpg",
#             "https://deadline.com/wp-content/uploads/2022/01/Bill-Maher-2021-HBO.jpeg",
#             "https://static01.nyt.com/images/2023/09/18/business/18strike-maher/18strike-maher-mediumSquareAt3X.jpg",
#             "https://www.rclco.com/wp-content/uploads/2023/11/Bill-Maher-Headshot-2023-Cropped.jpg",
#             "https://static01.nyt.com/images/2019/10/06/magazine/06mag-talk-01/06mag-talk-01-superJumbo.jpg",
#             "https://people.com/thmb/4cdiKi_Ay3Kh8nAwgmsQDK8bAuA=/4000x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(234x0:236x2)/bill_maher470-c3978280f1794e2eb96d507667dcd344.jpg",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2020/04/real_time_with_bill_maher.jpg?w=1296&h=730&crop=1",
#             "https://mediaassets.cbre.com/-/media/project/cbre/shared-site/headshots/b/i/l/bill-maher-752x.jpg",
#             "https://www.sky.com/assets2/real-time-with-bill-maher-panel-ccbacc2b.jpg",
#             "https://s.yimg.com/ny/api/res/1.2/YNLqG3jcObJi7U7o5sY6eQ--/YXBwaWQ9aGlnaGxhbmRlcjt3PTEyNDI7aD0xNDM4/https://media.zenfs.com/en/us_magazine_896/5f6f661f95aa4d2389c8e09c8441d720",
#             "https://deadline.com/wp-content/uploads/2021/05/Real-Time-with-Bill-Maher-2021.jpg?crop=54px%2C69px%2C1800px%2C1009px&resize=681%2C383",
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Bill_Maher_by_David_Shankbone.jpg/1200px-Bill_Maher_by_David_Shankbone.jpg",
#             "https://cloudfront-us-east-2.images.arcpublishing.com/reuters/U3SVXGO5VJJVRM2EG2S627APPQ.jpg",
#             "https://images.cdn.prd.api.discomax.com/2025/01/22/a4e4a418-64f1-3bbc-889e-64fda09e7fcb.jpeg?f=jpg&q=75&w=1280",
#             "https://ew.com/thmb/98MJix5zx5yyE145IXgEJT0LaaM=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/real-time-with-bill-maher-3-2000-9f7cf5d3098d462785975c9c8380fd75.jpg",
#             "https://cdn.britannica.com/27/157827-050-F7F43E50/Bill-Maher-2011.jpg?w=400&h=300&c=crop",
#             "https://www.nydailynews.com/wp-content/uploads/migration/2009/10/26/6ZB6ZJIAZAWW7T62PFIOAWAJGQ.jpg?w=535",
#             "https://www.politico.com/dims4/default/fbf2ff9/2147483647/legacy_thumbnail/1200x800%3E/quality/90/?url=http%3A%2F%2Fs3-origin-images.politico.com%2F2012%2F03%2F120316_bill_maher_facts_605_ap.jpg",
#             "https://media.vanityfair.com/photos/63d7e820269be31f5c11ceda/16:9/w_2000,h_1125,c_limit/Bill-Maher.jpg",
#             "https://static.independent.co.uk/2024/05/06/22/newFile-11.jpg",
#             "https://deadline.com/wp-content/uploads/2025/03/bill-maher-real-time-taylor-swift.jpg",
#             "https://aspenideasfestival.imgix.net/d466bfa5-365e-4238-bdf1-59921e474747/AIF24_Maher_Bill.jpeg?auto=compress%2Cformat&fit=min&fm=jpg&h=290&q=80&rect=652%2C19%2C3682%2C3686&w=290"
#           ]
#         }
#       ],
#       "clips": [
#         {
#           "characterName": "Bill Maher",
#           "videos": [
#             {
#               "url": "https://www.youtube.com/watch?v=tW0GowO_MaM",
#               "vidclips": [
#                 {
#                   "endTime": "00:05:59.94",
#                   "mute": True,
#                   "startTime": "00:05:51.94"
#                 }
#               ]
#             },
#             {
#               "url": "https://www.youtube.com/watch?v=yysKhJ1U-vM",
#               "vidclips": [
#                 {
#                   "endTime": "00:01:30.25",
#                   "mute": True,
#                   "startTime": "00:01:23.34"
#                 },
#                 {
#                   "endTime": "00:02:50.75",
#                   "mute": True,
#                   "startTime": "00:02:44.41"
#                 },
#                 {
#                   "endTime": "00:04:03.96",
#                   "mute": True,
#                   "startTime": "00:03:58.28"
#                 }
#               ]
#             }
#           ]
#         },
#         {
#           "characterName": "President Trump",
#           "videos": [
#             {
#               "url": "https://www.youtube.com/watch?v=O05fyieEkKQ",
#               "vidclips": [
#                 {
#                   "endTime": "00:00:41.58",
#                   "mute": True,
#                   "startTime": "00:00:34.72"
#                 },
#                 {
#                   "endTime": "00:01:15.10",
#                   "mute": True,
#                   "startTime": "00:01:05.24"
#                 }
#               ]
#             },
#             {
#               "url": "https://www.youtube.com/watch?v=nE1kGVAySBE",
#               "vidclips": [
#                 {
#                   "endTime": "00:00:08.54",
#                   "mute": True,
#                   "startTime": "00:00:01.44"
#                 },
#                 {
#                   "endTime": "00:00:28.90",
#                   "mute": True,
#                   "startTime": "00:00:22.37"
#                 },
#                 {
#                   "endTime": "00:01:28.46",
#                   "mute": True,
#                   "startTime": "00:01:16.04"
#                 }
#               ]
#             },
#             {
#               "url": "https://www.youtube.com/watch?v=6nEM62SJWO0",
#               "vidclips": [
#                 {
#                   "endTime": "00:03:38.29",
#                   "mute": True,
#                   "startTime": "00:02:49.49"
#                 },
#                 {
#                   "endTime": "00:07:19.42",
#                   "mute": True,
#                   "startTime": "00:07:13.25"
#                 }
#               ]
#             },
#             {
#               "url": "https://www.youtube.com/watch?v=5408ZhdP1OQ",
#               "vidclips": [
#                 {
#                   "endTime": "00:00:12.75",
#                   "mute": True,
#                   "startTime": "00:00:07.63"
#                 }
#               ]
#             }
#           ]
#         }
#       ],
#       "section_characters": [
#         "Bill Maher",
#         "President Trump"
#       ],
#       "segment": [
#         {
#           "character": "Bill Maher",
#           "text": "Quite surprising, isn't it?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "I get it, California is always at the forefront, pioneering trends that others might adopt, but here's Maher wondering, \"Is this the traditional stance of the Democratic party on immigration?\" Immigration has long been a topic both parties engaged with."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Consider this, while we all support spreading the American Dream, shouldn't it ideally be reserved for Americans?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Maher is skeptically lifting an eyebrow, seemingly questioning whether the guidelines have perhaps shifted too significantly."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "For Bill, the pieces just aren't adding up."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "He’s not against aiding others, yet he's delving deep, addressing the hard-hitting questions."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Gender Discussions Next up, the ongoing debate surrounding gender inclusion in sports."
#         },
#         {
#           "character": "President Trump",
#           "text": "President Trump has introduced an executive order to prevent boys from entering girls’ sports competitions."
#         },
#         {
#           "character": "President Trump",
#           "text": "The response was mixed, some applauded, others frowned upon it."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "But Maher?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "He's jumping into the fray, blending his trademark humor with incisive commentary."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Instead of taking a stand, you know what he does..?"
#         },
#         {
#           "character": "Bill Maher",
#           "tag": "sports",
#           "text": "[CLIP: President Trump this week just did an executive order saying that boys can no longer play in girls' sports. That's it. That's it. Ladies, isn't it ironic that the biggest protector of women turned out to be the pussy grabber? Huh? Maybe the guy who grabs it knows how to protect it the best. Think about that. Think about that. ]"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Maher is really targeting the paradox in the situation."
#         },
#         {
#           "character": "President Trump",
#           "text": "It's ironic how Trump is viewed as a defender of women, considering his notorious history with scandals."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Provoking Thoughts What I find fascinating about Maher is his approach."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "It goes beyond just irritation or humor."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "He genuinely stimulates reflection on cultural standards."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Meanwhile, he includes everyone in the humor, ensuring they stick around for the inevitable tweeting frenzy that follows."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "And guess what?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "There’s even more from Maher."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "His latest critiques of progressives form quite the spectacle."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "This time around, he's exposing what he perceives as brainwashing among today's youth."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "He’s linking their backing of Hamas, a notably controversial opinion, with their anti-American views."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "You might not believe it, but Bill is advocating for a resurgence of patriotic values in education."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "He’s endorsing an educational system that fosters love for America, rather than disdain."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "In one of his classic criticisms, Maher has even spotlighted Senate Majority Leader Chuck Schumer’s concerns over anti-Semitism, an issue that is extremely relevant today."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Maher is echoing Schumer’s concerns about dangerous chants and their implications regarding erasure."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Watching this develop is quite alarming, to be honest."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Maher is prominently supporting, reinforcing the message that Schumer has established."
#         },
#         {
#           "character": "Bill Maher",
#           "tag": "administration",
#           "text": "[CLIP: Under his administration, his next one, schools will teach students to love their country, not to hate their country, like they're taught now. I got to say, this one doesn't bother me so much, because I think that, I mean, this is what I see when I see kids demonstrating these progressives, demonstrating for Hamas, the most illiberal people in the world, that, oh, good, we're going to give America its comeuppance, asshole America. This is, they kind of have been indoctrinated this way. And Chuck Schumer made a speech this week. He said anti-Semitism is a five-alarm fire that must be extinguished. This is the highest-ranking Jewish politician we've ever had in this country. He talked about when Jewish people hear chants like, from the river to the sea, you understand that means wipe us out by any means necessary. Vulnerabilities in the party, James. Democratic Contradictions ]"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "In the midst of all the chaos, there's a palpable sense of exasperation within the Democratic party."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Isn't it astonishing how Democrats often entangle themselves in their own contradictions?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Take Maher’s razor-sharp commentary, for example—it’s utterly mind-blowing."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Does this sound familiar?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Democrats seem to find themselves navigating challenges in a convoluted and sometimes outright illogical manner."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "And there’s Maher, schooling everyone on how to tackle issues directly!"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "It's clear why his straightforward and candid insights are gaining attention."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "So, I have to wonder... what are your thoughts?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Are you stunned by Maher’s bold style, or do you see the cleverness in his remarks?"
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Leave your thoughts below and share your views on Maher's characterization of education as a \"black hole\" or his probing questions about who really benefits from the American Dream."
#         },
#         {
#           "character": "Bill Maher",
#           "text": "Eagerly awaiting your input!"
#         }
#       ],
#       "section_images": [
#         {
#           "character": "Bill Maher",
#           "images": [
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/BillMaherSept10.jpg/1200px-BillMaherSept10.jpg",
#             "https://static.hbo.com/2024-12/real-time-with-bill-maher-s23-ka-1920.jpg",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2024/05/1200/675/BillMaher53124.jpg?ve=1&tl=1",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2025/03/1200/675/mahermuskchainsaw2.jpg?ve=1&tl=1",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2022/05/1200/675/d849cd06-Maher.jpg?ve=1&tl=1",
#             "https://thecomedystore.com/wp-content/uploads/2023/06/e5984e4c255ee9ceee6f7c2fde95508a.jpg",
#             "https://edge.dailyvoice.com/upload/FotoJet_HnzrXOO.jpg?width=720",
#             "https://static.hbo.com/2022-04/bill-maher-adulting-ka-1920.jpg?w=1200",
#             "https://www.nj.com/resizer/v2/N72EFUO2I5A5PKGQ4WNP55CRJU.jpg?auth=4c2b5eb7ee833d3e224fb747b26cc422ee5cf8f00af6479f1cfae7380dcc2b5f&width=500&quality=90",
#             "https://cdn.choosechicago.com/uploads/2024/10/Bill-Maher-BIC-1280x820-1.jpg",
#             "https://media.gq.com/photos/5c7571166a6e6372b18b6e67/16:9/w_2560%2Cc_limit/Bill-Maher-Retire-Bitch-GQ-2019-022619.jpg",
#             "https://variety.com/wp-content/uploads/2025/03/don.png?w=1000&h=667&crop=1",
#             "https://i0.wp.com/www.iconvsicon.com/wp-content/uploads/2025/01/billmaher2.webp?resize=800%2C1185&ssl=1",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2021/09/bill-maher-andrew-yang-H-2021.jpg",
#             "https://latenighter.com/wp-content/uploads/2024/07/TSDSARA_EC006.jpg",
#             "https://m.media-amazon.com/images/M/MV5BNTQ4MzgwMTcxMl5BMl5BanBnXkFtZTcwNjAyNzQzMg@@._V1_FMjpg_UX1000_.jpg",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2025/03/dana-carvey-bill-maher.jpg?w=1296",
#             "https://static.wikia.nocookie.net/usam/images/9/99/Bill_Maher%27s_Official_Portrait.png/revision/latest/scale-to-width-down/1170?cb=20221225200507",
#             "https://www.thekeyexecutives.com/wp-content/uploads/2021/08/Bill-Maher.jpg",
#             "https://ffrf.org/wp-content/uploads/1980/01/90c519c0607749e3c2d669154a64428f.jpg",
#             "https://variety.com/wp-content/uploads/2017/01/real-time-with-bill-maher.jpg?w=1000&h=562&crop=1",
#             "https://static.cdn.turner.com/styles/header_image_1500x500_cropped/s3/2025-01/realtimewithbillmaher_s23_ho_ka_27x40_publicity_0.jpg?h=cfa352ee&itok=U7X-ziHP",
#             "https://www.apbspeakers.com/media/cirhnbev/maher_bill.jpg?rxy=0.40666666666666668,0.38016528925619836&height=775&quality=70&v=1daf95534af6fe0)",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2024/04/1200/675/MaherTrump.jpg?ve=1&tl=1",
#             "https://people.com/thmb/CSefpKe40XHzdl7qwhpoQyWtqB4=/4000x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(535x0:537x2)/bill-maher--Noor-Alfallah-1-100924-a7a0c0ab3b4f46dd8606832a1990eb63.jpg",
#             "https://media.zenfs.com/en/parade_250/9d6e4237175a3d23199b5a3c8499c057",
#             "https://cbsaustin.com/resources/media2/16x9/full/1015/center/80/09186cb0-09e0-4fb3-9b25-5f9347afec0e-large16x9_AP21134082736586.jpg",
#             "https://external-preview.redd.it/its-an-honor-bill-maher-explains-why-hes-going-to-meet-soon-v0-s3QJw9zggIIDHmroTqYVCLKppOMeZGTkVvIqoChe7cs.jpg?width=640&crop=smart&auto=webp&s=1f860ffe64573313f84cdeb5e8669d3f11ef7860",
#             "https://m.media-amazon.com/images/S/pv-target-images/c6b83ca34a7c8ddd424a0ccb97d2cfc1bec750906d05f50ac4666dd7a7daf3d2._SX1080_FMjpg_.jpg",
#             "https://static.hbo.com/content/dam/hbodata/series/real-time-with-bill-maher/episodes/15/800603_RTWBM_S15_L0_3000x1688.jpg",
#             "https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=1154787349348133",
#             "https://deadline.com/wp-content/uploads/2025/03/Bill-Maher-PA-Gov.-Josh-Shapiro.jpg",
#             "https://nypost.com/wp-content/uploads/sites/2/2025/03/bill-maher-visits-megyn-kelly-100868663.jpg?quality=75&strip=all&w=1200",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2024/03/Bill-Maher-getty-H-2024.jpg?w=1296&h=730&crop=1",
#             "https://kleinhansbuffalo.org/wp-content/uploads/sites/2/2022/03/Bill-Maher-credit-David-Becker-WireImage-11.25.13-scaled.jpg",
#             "https://variety.com/wp-content/uploads/2024/11/GettyImages-2153895970-e1730996650378.jpg?w=1000&h=667&crop=1",
#             "https://dw-wp-production.imgix.net/2025/03/GettyImages-2202961976-scaled.jpg?ar=16:9&fit=crop&crop=faces&fm=webp",
#             "https://ntvb.tmsimg.com/assets/p185450_b_h8_ab.jpg?w=1280&h=720",
#             "https://people.com/thmb/QgC19bpHBlHGR-CodUM6tV1T5hQ=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(999x0:1001x2)/bill-maher-1d579295ab1e4fa7b361b5dca2627685.jpg",
#             "https://i0.wp.com/www.nydailynews.com/wp-content/uploads/2024/10/241022_maher2.jpg?fit=620%2C9999px&ssl=1",
#             "https://m.media-amazon.com/images/M/MV5BMTUxNTY4NzIwMl5BMl5BanBnXkFtZTgwNDIxMTQ5NjE@._V1_.jpg",
#             "https://d.newsweek.com/en/full/2603083/bill-maher.jpg?w=1600&h=1200&q=88&f=a39cb4f94b906da2034e9d784c6d2941",
#             "https://i.ytimg.com/vi/MPb1VNt2EOo/maxresdefault.jpg",
#             "https://i.ytimg.com/vi/tbWNXam34wI/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLBBxIuMdjc0wpTd5oPECiSCdNUOZA",
#             "https://lookaside.instagram.com/seo/google_widget/crawler/?media_id=3573051297643545119",
#             "https://media.npr.org/assets/img/2023/09/14/gettyimages-866151214_wide-76eb9e6773132611d45af8bb32fa0dd16088d2a9.jpg?s=1100&c=50&f=jpeg",
#             "https://d28hgpri8am2if.cloudfront.net/book_images/onix/cvr9781668051351/what-this-comedian-said-will-shock-you-9781668051351_lg.jpg",
#             "https://heroicmedia.org/wp-content/uploads/2024/04/bill-maher-scaled.webp",
#             "https://pbs.twimg.com/ext_tw_video_thumb/1903280147840765952/pu/img/yWyF4WccypovuG3k.jpg:large",
#             "https://cdnph.upi.com/svc/sv/upi/9221730999107/2024/1/78852f5b1f6cde4303514e743e208fdd/Bill-Maher-comedy-special-Is-Anyone-Else-Seeing-This-coming-in-January.jpg",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2024/05/BillMaherGettyImages-2074809903.jpg?w=1296&h=730&crop=1",
#             "https://www.advocate.com/media-library/trans-comic-on-bill-maher-grow-up-and-go-away.jpg?id=32678112&width=1245&height=700&coordinates=0%2C47%2C0%2C96",
#             "https://m.media-amazon.com/images/S/pv-target-images/eddcbb81901ca6fc2fb76cee0dc3d67fc8eefb2b64cd18d719d60a26fdaee626.jpg",
#             "https://lookaside.instagram.com/seo/google_widget/crawler/?media_id=3594122346437716064",
#             "https://www.pabsttheatergroup.com/assets/img/2024.07.14-R-Bill-Maher500x500-Web-Thumbnail-d8b5d09e68.png",
#             "https://i.guim.co.uk/img/media/51c8821a9c1cfdbff33612063e293ba7432a591b/479_69_2515_1509/master/2515.jpg?width=465&dpr=1&s=none&crop=none",
#             "https://variety.com/wp-content/uploads/2025/03/maronmaher-split.jpg?w=1000&h=666&crop=1",
#             "https://i.ytimg.com/vi/d6qhapDVUNc/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLDufefHZ79MVhYlabSFMw_gQP8dFA",
#             "https://www.portland5.com/sites/default/files/styles/event_square_large/public/events/2022/11/03/Static_Digital_438x400_BillMaher_2023_Regional_ArleneSchnitzerConcertHall_0402.jpg?itok=D04tgOjq",
#             "https://9b16f79ca967fd0708d1-2713572fef44aa49ec323e813b06d2d9.ssl.cf2.rackcdn.com/1140x_a10-7_cTC/author-and-comedian-bill-maher-1720109364.jpg",
#             "https://nypost.com/wp-content/uploads/sites/2/2025/03/bill-maher-100922151.jpg?quality=75&strip=all&w=1024",
#             "https://www.catholicleague.org/wp-content/uploads/2015/06/BillMaher.jpg",
#             "https://www.thenation.com/wp-content/uploads/2015/03/bill_maher_ap_img.jpg",
#             "https://resizing.flixster.com/-XZAfHZM39UwaGJIFWKAE8fS0ak=/v3/t/assets/1076_v9_bb.jpg",
#             "https://cloudfront-us-east-1.images.arcpublishing.com/advancelocal/MWFO34OGWNBPZHG34KPFYANLU4.jpg",
#             "https://deadline.com/wp-content/uploads/2022/01/Bill-Maher-2021-HBO.jpeg",
#             "https://i.scdn.co/image/ab67656300005f1f1ab33a96d6e81f73976168ef",
#             "https://m.media-amazon.com/images/I/91jgsqDDw3L._AC_UF894,1000_QL80_.jpg",
#             "https://static01.nyt.com/images/2023/09/18/business/18strike-maher/18strike-maher-mediumSquareAt3X.jpg",
#             "https://m.media-amazon.com/images/M/MV5BMjU3OTc0NDQtNzIwMi00YzY0LTkyODctYTFhOGVhYTc1YzdhXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
#             "https://www.rclco.com/wp-content/uploads/2023/11/Bill-Maher-Headshot-2023-Cropped.jpg",
#             "https://i.ytimg.com/vi/V__wp4xvTjI/maxresdefault.jpg",
#             "https://static01.nyt.com/images/2019/10/06/magazine/06mag-talk-01/06mag-talk-01-superJumbo.jpg",
#             "https://people.com/thmb/4cdiKi_Ay3Kh8nAwgmsQDK8bAuA=/4000x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(234x0:236x2)/bill_maher470-c3978280f1794e2eb96d507667dcd344.jpg",
#             "https://www.hollywoodreporter.com/wp-content/uploads/2020/04/real_time_with_bill_maher.jpg?w=1296&h=730&crop=1",
#             "https://blog-admin.siriusxm.com/wp-content/uploads/2024/05/bill-maher-megyn-kelly3.jpg?w=885",
#             "https://mediaassets.cbre.com/-/media/project/cbre/shared-site/headshots/b/i/l/bill-maher-752x.jpg",
#             "https://www.usatoday.com/gcdn/-mm-/7268d9548b6d8c5c2f0290034b4c365680fd404b/c=0-107-1422-910/local/-/media/2017/06/10/USATODAY/USATODAY/636326590727276870-AP-TV-Bill-Maher.jpg?width=660&height=371&fit=crop&format=pjpg&auto=webp",
#             "https://www.sky.com/assets2/real-time-with-bill-maher-panel-ccbacc2b.jpg",
#             "https://thehill.com/wp-content/uploads/sites/2/2024/06/Joe-Biden-Bill-Maher-06.29.png?strip=1",
#             "https://images.nymag.com/news/features/maher120416_1_560.jpg",
#             "https://s.yimg.com/ny/api/res/1.2/YNLqG3jcObJi7U7o5sY6eQ--/YXBwaWQ9aGlnaGxhbmRlcjt3PTEyNDI7aD0xNDM4/https://media.zenfs.com/en/us_magazine_896/5f6f661f95aa4d2389c8e09c8441d720",
#             "https://i.ytimg.com/vi/V7R9h-SCX68/sddefault.jpg",
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Bill_Maher_by_David_Shankbone.jpg/1200px-Bill_Maher_by_David_Shankbone.jpg",
#             "https://deadline.com/wp-content/uploads/2021/05/Real-Time-with-Bill-Maher-2021.jpg?crop=54px%2C69px%2C1800px%2C1009px&resize=681%2C383",
#             "https://cloudfront-us-east-2.images.arcpublishing.com/reuters/U3SVXGO5VJJVRM2EG2S627APPQ.jpg",
#             "https://images.cdn.prd.api.discomax.com/2025/01/22/a4e4a418-64f1-3bbc-889e-64fda09e7fcb.jpeg?f=jpg&q=75&w=1280",
#             "https://pbs.twimg.com/profile_images/1744432463890567168/2l7kNwU4_400x400.jpg",
#             "https://www.catholicleague.org/wp-content/uploads/2023/01/Maher-web-pic.png",
#             "https://m.media-amazon.com/images/I/71o6EgaoYLL._AC_UF894,1000_QL80_.jpg",
#             "https://ew.com/thmb/98MJix5zx5yyE145IXgEJT0LaaM=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/real-time-with-bill-maher-3-2000-9f7cf5d3098d462785975c9c8380fd75.jpg",
#             "https://cdn.britannica.com/27/157827-050-F7F43E50/Bill-Maher-2011.jpg?w=400&h=300&c=crop",
#             "https://www.nydailynews.com/wp-content/uploads/migration/2009/10/26/6ZB6ZJIAZAWW7T62PFIOAWAJGQ.jpg?w=535",
#             "https://opinion-images.wsj.net/im-86981325/?size=1.5",
#             "https://www.politico.com/dims4/default/fbf2ff9/2147483647/legacy_thumbnail/1200x800%3E/quality/90/?url=http%3A%2F%2Fs3-origin-images.politico.com%2F2012%2F03%2F120316_bill_maher_facts_605_ap.jpg",
#             "https://media.vanityfair.com/photos/63d7e820269be31f5c11ceda/16:9/w_2000,h_1125,c_limit/Bill-Maher.jpg",
#             "https://static.independent.co.uk/2024/05/06/22/newFile-11.jpg",
#             "https://deadline.com/wp-content/uploads/2025/03/bill-maher-real-time-taylor-swift.jpg",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2024/05/maher-1024x576.jpg",
#             "https://aspenideasfestival.imgix.net/d466bfa5-365e-4238-bdf1-59921e474747/AIF24_Maher_Bill.jpeg?auto=compress%2Cformat&fit=min&fm=jpg&h=290&q=80&rect=652%2C19%2C3682%2C3686&w=290"
#           ]
#         },
#         {
#           "character": "President Trump",
#           "images": [
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/TrumpPortrait.jpg/800px-TrumpPortrait.jpg",
#             "https://bidenwhitehouse.archives.gov/wp-content/uploads/2021/01/45_donald_trump.jpg",
#             "https://i.insider.com/6797aeb3196626c409851836?width=1200&format=jpeg",
#             "https://www.usda.gov/sites/default/files/styles/two_column_header_image_1320x888/public/sec-rollins-nat-ag-day-pres-trump.jpg?h=c6980913&itok=62Vk6pFG",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2024/11/dt-winner-1024x585.jpg",
#             "https://d.newsweek.com/en/full/2613100/elon-musk-donald-trump.jpg?w=1200&f=f32e1a1e27eb66aad8e734e63e217995",
#             "https://cdn.britannica.com/21/197021-050-D0AC3E38/portrait-Donald-Trump.jpg",
#             "https://nypost.com/wp-content/uploads/sites/2/2025/03/united-states-president-donald-trump-100413395.jpg?quality=75&strip=all&w=744",
#             "https://s.abcnews.com/images/US/donald-trump-10-ap-gmh-241106_1730906289206_hpMain_16x9_1600.jpg",
#             "https://i.natgeofe.com/k/5e4ea67e-2219-4de4-9240-2992faef0cb6/trump-portrait.jpg",
#             "https://media.newyorker.com/photos/67902dac62c7d191667ff4ad/master/w_2560%2Cc_limit/Gersen-TrumpSignings-Mills.jpg",
#             "https://static01.nyt.com/newsgraphics/2019/08/01/candidate-pages/3b31eab6a3fd70444f76f133924ae4317567b2b5/trump.jpg",
#             "https://idahocapitalsun.com/wp-content/uploads/2024/11/trumpnov13-scaled.jpg",
#             "https://c.files.bbci.co.uk/4616/live/d710a6a0-d4b0-11ef-9fd6-0be88a764111.jpg",
#             "https://npr.brightspotcdn.com/dims3/default/strip/false/crop/1024x732+0+0/resize/1100/quality/50/format/jpeg/?url=http%3A%2F%2Fnpr-brightspot.s3.amazonaws.com%2F00%2Fb9%2F4c96b78f469a868605baad1a185c%2Fgettyimages-2203857531.jpg",
#             "https://ru.usembassy.gov/wp-content/uploads/sites/138/AP25020618863748-scaled.jpg",
#             "https://www.reuters.com/resizer/v2/UUYZET5LUJINVGAODHS5PI2XNQ.jpg?auth=9cd79b0ffff001238d732400bd2701cb088bd0df4c05413ea801140e30d9d01c&width=5198&quality=80",
#             "https://media.newyorker.com/photos/669324fd14c38da9ac7b7815/4:3/w_2281,h_1711,c_limit/Trump-shot-AP24195851874181.jpg",
#             "https://media.cnn.com/api/v1/images/stellar/prod/gettyimages-2205875388.jpg?q=w_1110,c_fill",
#             "https://static.scientificamerican.com/dam/m/530f6a83ec06a4d4/original/Trump_signs_executive_orders.jpg?m=1737561002.57&crop=16%3A9%2Csmart&w=1920",
#             "https://media.vanityfair.com/photos/6371861f57377e33b6f57199/master/w_2560%2Cc_limit/1431869535",
#             "https://news.berkeley.edu/wp-content/uploads/2025/03/APTOPIX_Trump_Zelenskyy_25059642193910-scaled.jpg",
#             "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Donald_Trump_%288567815294%29_%28cropped%29.jpg/1200px-Donald_Trump_%288567815294%29_%28cropped%29.jpg",
#             "https://www.reuters.com/resizer/v2/KSPOO6STX5LFVNXDWHOA7WXAP4.jpg?auth=24b60ea0f5a37148dd3742ab6080f7756ba2000a574025ae98073eea7ebd5015&width=1080&quality=80",
#             "https://people.com/thmb/Ur9UpqkezoBaDUb7y-xy6fGJA8Q=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(753x318:755x320)/donald-trump-shut-up-about-egg-prices-031125-5fedea9793b24175a12ddc5db2984d82.jpg",
#             "https://si.usembassy.gov/wp-content/uploads/sites/113/2025/02/trump_profile_smithsonian-.jpeg",
#             "https://ca-times.brightspotcdn.com/dims4/default/28245b3/2147483647/strip/true/crop/3000x2000+0+0/resize/1200x800!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2F71%2Ff0%2F6aba34814415ad71445b4e70b94f%2Feeuu-elecciones-opiniones-de-votantes-00030.jpg",
#             "https://i.abcnewsfe.com/a/0f6aa83e-b2e3-4d9d-bae6-681d4a8585f3/apprentice-2-rt-er-250219_1740001709973_hpMain_9x16.jpg?w=992",
#             "https://defensescoop.com/wp-content/uploads/sites/8/2024/12/Trump_d7ed75.jpg?w=1013",
#             "https://www.usatoday.com/gcdn/authoring/authoring-images/2025/01/30/USAT/78061580007-trump.jpg?crop=2833,1594,x0,y0&width=660&height=371&format=pjpg&auto=webp",
#             "https://www.pewresearch.org/wp-content/uploads/sites/20/2025/03/SR_25.03.03_sotu_feature.jpg?w=640",
#             "https://npr.brightspotcdn.com/dims3/default/strip/false/crop/5236x3491+0+0/resize/1100/quality/50/format/jpeg/?url=http%3A%2F%2Fnpr-brightspot.s3.amazonaws.com%2F75%2F46%2F50d8174944668a6411b3b1fb4003%2Fap25064161390673.jpg",
#             "https://time.com/ayit/images/7200212/cover.png",
#             "https://snworksceo.imgix.net/dtc/df21c545-b4b8-4d85-ae96-507181573926.sized-1000x1000.jpg?w=1000&ar=16%3B9&fit=crop&crop=faces&facepad=3&auto=format",
#             "https://cdn.theatlantic.com/thumbor/fhigyO0dP3BlYDmx2H4g89kkA5o=/0x281:5574x3416/960x540/media/img/mt/2020/03/RTS37HPB-1/original.jpg",
#             "https://images.theconversation.com/files/643798/original/file-20250121-15-mhxlbo.jpg?ixlib=rb-4.1.0&rect=0%2C86%2C7185%2C4707&q=20&auto=format&w=320&fit=clip&dpr=2&usm=12&cs=strip",
#             "https://lailluminator.com/wp-content/uploads/2024/02/TrumpSC-scaled-1-1536x1024.jpg",
#             "https://m.media-amazon.com/images/I/41+6hVAAvkL._AC_UF894,1000_QL80_.jpg",
#             "https://dims.apnews.com/dims4/default/adb221b/2147483647/strip/true/crop/5716x3815+0+0/resize/599x400!/quality/90/?url=https%3A%2F%2Fassets.apnews.com%2F20%2F8c%2F0a9ef8a741a8bea73ae6619c58d7%2Fap25042756828063.jpg",
#             "https://platform.vox.com/wp-content/uploads/sites/2/2025/03/GettyImages-2203829700.jpg?quality=90&strip=all&crop=0%2C0%2C100%2C100&w=2400",
#             "https://d.newsweek.com/en/full/2572575/trump.jpg?w=1200&f=a8ce696455921d925bfaca5d9118b82d",
#             "https://www.bu.edu/files/2024/11/trump-research-feat.jpg",
#             "https://ca-times.brightspotcdn.com/dims4/default/abeeca8/2147483647/strip/true/crop/6000x4000+0+0/resize/1200x800!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2Fca%2F81%2F059f33084d28b73bcd0d100cef87%2Ftrump-61047.jpg",
#             "https://imageio.forbes.com/specials-images/imageserve/66e9ea2f67d3fefc9fbe1239/4x5-Donald-Trump-by-Spencer-Platt-Getty-Images/0x0.jpg?format=jpg&height=2401&width=1920",
#             "https://platform.vox.com/wp-content/uploads/sites/2/chorus/uploads/chorus_asset/file/25274440/1965959342.jpg?quality=90&strip=all&crop=0%2C0%2C100%2C100&w=2400",
#             "https://kansasreflector.com/wp-content/uploads/2024/07/GettyImages-2162619255-2048x1365-1-1536x1024.jpg",
#             "https://whyy.org/wp-content/uploads/2024/11/donald-trump-election-presidential-race-AP-110624.jpg",
#             "https://fortune.com/img-assets/wp-content/uploads/2025/02/GettyImages-2201780017-e1740583416448.jpg?w=1440&q=75",
#             "https://www.thinkglobalhealth.org/sites/default/files/styles/max_325x325/public/2025-01/2025-01-27T165747Z_1114190312_RC21ECAOF4QD_RTRMADP_3_USA-TRUMP-LGBT-LAWSUIT.JPG?itok=egu0Xugl",
#             "https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iCy6sk0oBAIE/v1/-1x-1.webp",
#             "https://michiganadvance.com/wp-content/uploads/2025/01/trumpjan20-1536x1024.jpg",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2025/01/2025-01-29T172751Z_1854301353_RC2VFCA9S0DW_RTRMADP_3_BRITAIN-TRUMP-COURT-1024x683.jpg",
#             "https://www.aljazeera.com/wp-content/uploads/2024/11/AP24309574257630-1730736613.jpg?resize=1920%2C1080",
#             "https://cdn.cfr.org/sites/default/files/styles/full_width_xl/public/image/2025/01/Trump%20Rubio.webp",
#             "https://media-cldnry.s-nbcnews.com/image/upload/t_fit-760w,f_auto,q_auto:best/rockcms/2025-03/250311-donald-trump-tesla-ew-444p-6d3252.jpg",
#             "https://s.france24.com/media/display/1c3a99ce-6556-11ed-9bd5-005056a90284/w:1280/p:16x9/2022-11-16T022244Z_543999449_RC2QMX95YN5H_RTRMADP_3_USA-ELECTION-TRUMP.JPG",
#             "https://a57.foxnews.com/static.foxnews.com/foxnews.com/content/uploads/2025/03/1200/675/trump-speech.jpg?ve=1&tl=1",
#             "https://whyy.org/wp-content/uploads/2025/02/trump-npr-021225-768x512.jpeg",
#             "https://www.science.org/do/10.1126/science.zxvum0n/full/_20250121_nid_trump_who-1737648919077.jpg",
#             "https://i.ytimg.com/vi/74fDmAt4Mp8/maxresdefault.jpg",
#             "https://www.thoughtco.com/thmb/OIQxajVv1AMskdgr21fgM-qpCOo=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-1163026796-ef8fdfa1a68a471d8bda5aeab73c98c0.jpg",
#             "https://media.cnn.com/api/v1/images/stellar/prod/gettyimages-2194383933.jpg?c=16x9&q=h_833,w_1480,c_fill",
#             "https://www.citizensforethics.org/wp-content/uploads/2020/09/50349050837_06318c5780_b-1200x675.jpg",
#             "https://www.brookings.edu/wp-content/uploads/2025/03/2025-03-05T141645Z_1497928456_MT1USATODAY25585055_RTRMADP_3_MAR-4-2025-WASHINGTON-DC-USA-PRESIDENT-DONALD-TRUMP-1.jpg?quality=75&w=1500",
#             "https://www.economist.com/cdn-cgi/image/width=1424,quality=80,format=auto/content-assets/images/20240608_USP002.jpg",
#             "https://c.files.bbci.co.uk/C51D/production/_133516405_updatedpromoimage.png",
#             "https://cdn.theatlantic.com/thumbor/5o08pfdzYI358Bx27p2MobrBek8=/0x0:3000x3750/648x810/media/img/2024/11/07/trump_normalize/original.jpg",
#             "https://www.aa.com.tr/uploads/userFiles/595e16c1-a428-468e-9249-631d5c6ef9ef/World-leaders-congratulate-Trump-after-his-election-victory-4.jpg",
#             "https://hips.hearstapps.com/hmg-prod/images/gettyimages-2194420718-67d9b4e326598.jpg?crop=0.646xw:0.969xh;0.148xw,0&resize=640:*",
#             "https://media-cldnry.s-nbcnews.com/image/upload/t_fit-1500w,f_auto,q_auto:best/rockcms/2023-09/230921-donald-trump-jm-1109-57b054.jpg",
#             "https://www.citizensforethics.org/wp-content/uploads/2021/02/46451130962_d151b497d6_b-1200x675.jpg",
#             "https://physicsworld.com/wp-content/uploads/2024/12/Trump-re-election-6-Nov-2024-small.jpg",
#             "https://cdn.abcotvs.com/dip/images/14890421_053024-cc-trump-ap-img.jpg",
#             "https://media.vanityfair.com/photos/672a497a98a8cfd30ca61fde/master/w_2560%2Cc_limit/fv1124-trump.jpg",
#             "https://i.ytimg.com/vi/E1oCr6apOKs/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLCIwgGMw24KuqA_G-pSyKXHoaK8Eg",
#             "https://assets.medpagetoday.net/media/images/111xxx/111859.jpg?width=0.8",
#             "https://people.com/thmb/IUi55iAYATKHqTaxTKnY8tPWCc8=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(749x191:751x193)/Donald-Trump-Grover-Cleveland-Tout-102824-d5e8b58f46e442879578752bcd467325.jpg",
#             "https://m.economictimes.com/thumb/msid-115025097,width-1600,height-900,resizemode-4,imgsize-45320/donald-trump-is-usa-president-elect-what-does-it-mean-for-the-world.jpg",
#             "https://npr.brightspotcdn.com/dims3/default/strip/false/crop/5816x3877+0+0/resize/1100/quality/50/format/jpeg/?url=http%3A%2F%2Fnpr-brightspot.s3.amazonaws.com%2F33%2Fef%2Fcad7a8bf4e899704c0c379f22653%2Fgettyimages-2203683508.jpg",
#             "https://media.defense.gov/2019/Aug/30/2002177620/1920/1080/0/190829-D-BN624-0255.JPG",
#             "https://www.aljazeera.com/wp-content/uploads/2024/12/2024-12-07T175425Z_1928241406_RC2FKBARY8Z2_RTRMADP_3_USA-TRUMP-FRANCE-1733734296.jpg?resize=770%2C513&quality=80",
#             "https://www.thirteen.org/wp-content/uploads/sites/2/2025/01/DonaldTrumpInauguration.jpg",
#             "http://media.newyorker.com/photos/5e6a47215a2694000896ab00/master/pass/Glasser-TrumpCoronavirusSpeech.jpg",
#             "https://bsmedia.business-standard.com/_media/bs/img/article/2024-11/15/full/1731662391-1183.jpg?im=FeatureCrop,size=(826,465)",
#             "https://images.wsj.net/im-82923306/?width=1280&height=1280",
#             "https://cdn.abcotvs.com/dip/images/15821493_trump-orders-img.png",
#             "https://www.brookings.edu/wp-content/uploads/2024/10/2024-09-11T054926Z_2092926145_MT1USATODAY24204474_RTRMADP_3_SEP-10-2024-PHILADELPHIA-PA-USA-FORMER-PRESIDENT-DONALD1.jpg?quality=75&w=1500",
#             "https://images.newrepublic.com/43817d344cc9aa0f15fd78e6c8349809bb904b6f.jpeg?auto=format&fit=crop&crop=faces&q=65&w=768&ar=3%3A2&ixlib=react-9.10.0",
#             "https://vn.usembassy.gov/wp-content/uploads/sites/124/2025/01/President-Trump-priorities.jpg",
#             "https://static.euronews.com/articles/stories/08/76/78/88/1200x675_cmsv2_4a4d8afe-2d30-5d5c-aa84-13546c1a64d6-8767888.jpg",
#             "https://fortune.com/img-assets/wp-content/uploads/2025/01/GettyImages-2194398603.jpg?w=1440&q=75",
#             "https://img-cdn.inc.com/image/upload/f_webp,q_auto,c_fit/vip/2024/10/donald-trump-wins-election-inc.jpg",
#             "https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=1132495238558163&get_thumbnail=1",
#             "https://static01.nyt.com/images/2024/11/05/multimedia/pol-trump-wins-hfo-new0101-qzgw/pol-trump-wins-hfo-new0101-qzgw-mediumSquareAt3X.jpg",
#             "https://www.americanprogress.org/wp-content/uploads/sites/2/2017/04/AP_17115159898594-PREPPED.jpg?w=1040",
#             "https://dobetter.esade.edu/sites/default/files/post/2024-10/victory-trump-EU-w.webp",
#             "https://s.france24.com/media/display/8eb0b174-9c4e-11ef-8886-005056bfb2b6/w:1280/p:16x9/main-nebuleuse-trump.jpg",
#             "https://d3i6fh83elv35t.cloudfront.net/static/2025/02/2025-02-24T174606Z_1203290860_RC251DAHQ3D9_RTRMADP_3_USA-FRANCE-1024x682.jpg",
#             "https://www.whitehouse.gov/wp-content/themes/whitehouse/assets/img/whitehouse-47-donald-j-trump-cover.png",
#             "https://variety.com/wp-content/uploads/2024/12/GettyImages-2188144000.jpg?w=1000&h=667&crop=1"
#           ]
#         }
#       ],
#       "section_videos": [
#         {
#           "Character": "Bill Maher",
#           "videos": [
#             {
#               "title": "Main Video",
#               "url": "https://youtu.be/vRz47H4iFv4"
#             },
#             {
#               "title": "Real Time with Bill Maher: Ann Coulter on Immigration  (HBO)",
#               "url": "https://www.youtube.com/watch?v=tW0GowO_MaM"
#             },
#             {
#               "title": "Bill Maher on democrat immigration policies “it’s going to get the f***d on election day” #usa",
#               "url": "https://www.youtube.com/watch?v=afsmaxGElYI"
#             },
#             {
#               "title": "Bill Maher exposes how liberal hypocrisy on illegal immigration is a “disaster” for Democrats",
#               "url": "https://www.youtube.com/watch?v=29zTydu99-U"
#             },
#             {
#               "title": "Bill Maher TAKES ON Trans Athletes, Gender Affirming Surgery In DEBATE With Neil Degrasse Tyson",
#               "url": "https://www.youtube.com/watch?v=lagPV9CJNeI"
#             },
#             {
#               "title": "Bill Maher EVISCERATES Neil deGrasse Tyson On Transgender Athletes CONTROVERSY",
#               "url": "https://www.youtube.com/watch?v=QNePuvC3R4I"
#             },
#             {
#               "title": "MUST WATCH: Katie Porter, Piers Morgan, Bill Maher Debate Transgender Athletes In School Sports",
#               "url": "https://www.youtube.com/watch?v=wyIqL2Jh8xk"
#             },
#             {
#               "title": "New Rule: Progressophobia | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=fB9KVYAdYwg"
#             },
#             {
#               "title": "New Rule: Tough Love Dems | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=BtCK-dMb-F8"
#             },
#             {
#               "title": "New Rule: A Woke Revolution | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=yysKhJ1U-vM"
#             },
#             {
#               "title": "She SCHOOLS Bill Maher on Tariffs",
#               "url": "https://www.youtube.com/watch?v=i4aB10VNxMM"
#             },
#             {
#               "title": "New Rule: Fetish Patriotism | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=l6PT3RqJhe0"
#             },
#             {
#               "title": "Bill Maher Defends Trumps Answer To Education",
#               "url": "https://www.youtube.com/watch?v=j0MKt-PfY6I"
#             },
#             {
#               "title": "Sen. Chuck Schumer (D-NY) Confronted With Dem Hypocrisy On Bill Maher",
#               "url": "https://www.youtube.com/watch?v=wD07Brs-LuY"
#             },
#             {
#               "title": "Schumer slams Trump for tolerating antisemitism: ‘He’s not protecting Jewish people’",
#               "url": "https://www.youtube.com/watch?v=BP3MI6uA1nk"
#             },
#             {
#               "title": "Sen. Chuck Schumer Defends His Decision To Support Republican Spending Bill",
#               "url": "https://www.youtube.com/watch?v=7kpiiBdlUNQ"
#             }
#           ]
#         },
#         {
#           "Character": "President Trump",
#           "videos": [
#             {
#               "title": "Immigration Crisis: Trump administration continue deportations despite judge orders",
#               "url": "https://www.youtube.com/watch?v=O05fyieEkKQ"
#             },
#             {
#               "title": "Border crisis: Trump admin deports Venezuelan migrants to El Salvador | LiveNOW from FOX",
#               "url": "https://www.youtube.com/watch?v=zomXav3N7uw"
#             },
#             {
#               "title": "Trump administration deports hundreds of migrants despite judge's order stopping removal",
#               "url": "https://www.youtube.com/watch?v=t-cJsA8bAbc"
#             },
#             {
#               "title": "Heated Exchange Between President Trump and Gov. Janet Mills (D-ME) over Transgender Women in Sports",
#               "url": "https://www.youtube.com/watch?v=nE1kGVAySBE"
#             },
#             {
#               "title": "Donald Trump Blocks Transgender Athletes From Women’s Sports | 10 News First",
#               "url": "https://www.youtube.com/watch?v=64cJbCbOeCo"
#             },
#             {
#               "title": "Trump signs executive order banning trans women from women's sports",
#               "url": "https://www.youtube.com/watch?v=EYIE2JKmhKA"
#             },
#             {
#               "title": "LIVE: Pres. Trump signing executive order on trans athletes in women's sports",
#               "url": "https://www.youtube.com/watch?v=bCBvdSFC2Ps"
#             },
#             {
#               "title": "Trump signs executive order protecting women's sports",
#               "url": "https://www.youtube.com/watch?v=6nEM62SJWO0"
#             },
#             {
#               "title": "President Trump signs executive order banning transgender athletes from women's sports",
#               "url": "https://www.youtube.com/watch?v=5408ZhdP1OQ"
#             },
#             {
#               "title": "Free speech in Trump's America: Drawing up the battlelines • FRANCE 24 English",
#               "url": "https://www.youtube.com/watch?v=8w04_NU5JtA"
#             },
#             {
#               "title": "Donald Trump attacks 'woke' ideology in inauguration speech",
#               "url": "https://www.youtube.com/watch?v=AXLYiZ7RIig"
#             },
#             {
#               "title": "'We're basically sleeping on the floor': Veteran speaks about Trump's cuts",
#               "url": "https://www.youtube.com/watch?v=jtsWBi5GTKU"
#             },
#             {
#               "title": "Trump at National Archives: 'Patriotic Education' Must Be Restored to Schools",
#               "url": "https://www.youtube.com/watch?v=up2pzuE32tA"
#             },
#             {
#               "title": "Teachers’ union president spells out impact of Trump’s Education Department cuts",
#               "url": "https://www.youtube.com/watch?v=ZrNpdzBZBwQ"
#             },
#             {
#               "title": "Trump Plans to Promote 'Patriotic Education'",
#               "url": "https://www.youtube.com/watch?v=M8W57sCPApg"
#             }
#           ]
#         },
#         {
#           "Character": "Clip",
#           "videos": [
#             {
#               "title": "New Rule: America's Whore Complex | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=5coqEzHjQ9U"
#             },
#             {
#               "title": "New Rule: Political Firestorm | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=C5S8rhNCBnc"
#             },
#             {
#               "title": "Bill Maher: It’s not that I’ve gotten old. It’s that your ideas are stupid",
#               "url": "https://www.youtube.com/watch?v=Sh-57IbGUnQ"
#             },
#             {
#               "title": "Bill Maher exposes how liberal hypocrisy on illegal immigration is a “disaster” for Democrats",
#               "url": "https://www.youtube.com/watch?v=29zTydu99-U"
#             },
#             {
#               "title": "Bill Maher on democrat immigration policies “it’s going to get the f***d on election day” #usa",
#               "url": "https://www.youtube.com/watch?v=afsmaxGElYI"
#             },
#             {
#               "title": "New Rule: Identity Crisis | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=5Gb5MalUjvI"
#             },
#             {
#               "title": "New Rule: Along for the Pride | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=mMBzfUj5zsg"
#             },
#             {
#               "title": "New Rule: Gender Apartheid | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=uRzv0HgatRc"
#             },
#             {
#               "title": "Jon Lovett Debates Bill Maher on Transgender Rights",
#               "url": "https://www.youtube.com/watch?v=JEKyn9NL46U"
#             },
#             {
#               "title": "Bill Maher Urges Dems To Ignore Progressives",
#               "url": "https://www.youtube.com/watch?v=2IEEyH3_QmM"
#             },
#             {
#               "title": "New Rule: Progressophobia | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=fB9KVYAdYwg"
#             },
#             {
#               "title": "New Rule: Tough Love Dems | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=BtCK-dMb-F8"
#             },
#             {
#               "title": "\"Wipe It Out!\" - Glenn Beck FIRES BACK As Bill Maher Backs Trump's Department Of Education Shut Down",
#               "url": "https://www.youtube.com/watch?v=BU184HH5Byc"
#             },
#             {
#               "title": "How Bill Maher defines 'woke'",
#               "url": "https://www.youtube.com/watch?v=tzwC-10O0cw"
#             },
#             {
#               "title": "New Rule: Patriotic Privilege | Real Time with Bill Maher (HBO)",
#               "url": "https://www.youtube.com/watch?v=0LWe2L-k67o"
#             }
#           ]
#         }
#       ]
#     },
#     "conclusion": {
#       "section": "",
#       "section_script": "",
#       "imageData": [],
#       "clips": [],
#       "section_characters": [],
#       "segment": [],
#       "section_images": [],
#       "section_videos": []
#     }
#   },
# "asset_folder": "20250311_055831-20250311_061125",
# "video_path": "final_video_20250311_062454.mp4"
# }

# if __name__ == "__main__":
#     import json
    
#     print("🚀 Running Full Pipeline...\n")
#     final_result = asyncio.run(full_pipeline(video_data, niche='default'))

    # Preview output
    # print(final_result)
    # intro = final_result["sections"].get("intro", {})
    # for seg in intro.get("segment", [])[:20]:  # Limit to first 5
    #     print(f"[{seg['character']}]: {seg['text']}\n")

    # print("\n✅ SAMPLE OUTPUT (BODY SECTION):")
    # intro = final_result["sections"].get("body", {})
    # for seg in intro.get("segment", [])[:20]:  # Limit to first 5
    #     print(f"[{seg['character']}]: {seg['text']}\n")



#     print("✅ Full output saved to final_output.json")


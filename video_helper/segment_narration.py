import os
import logging
import openai
import re
import difflib
import traceback
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def estimate_segments(narration_text):
    """Split the narration text into segments, handling various punctuation."""
    if not narration_text or not isinstance(narration_text, str):
        return []
    
    # Protect ellipses and common abbreviations
    protected_text = re.sub(r'(\w\.\w\.)', r'\1PROTECTED', narration_text)
    protected_text = re.sub(r'(Mr\.|Mrs\.|Dr\.|etc\.|i\.e\.|e\.g\.)', r'\1PROTECTED', protected_text)
    protected_text = re.sub(r'\.\.\.', 'ELLIPSIS', protected_text)
    
    # Split by sentence-ending punctuation
    raw_segments = re.split(r'(?<!\w\.\w\.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\:)\s', protected_text)
    
    # Restore protected portions and clean up
    segments = []
    for segment in raw_segments:
        if segment.strip():
            cleaned = segment.replace('PROTECTED', '').replace('ELLIPSIS', '...')
            # Ensure segment ends with punctuation
            if not cleaned.endswith(('.', '!', '?', ':')):
                cleaned += '.'
            segments.append(cleaned.strip())
    
    return segments

def extract_celebrity_names(text):
    """Extract potential celebrity names from text using NER via the API."""
    try:
        system_prompt = (
            "You are an expert in identifying celebrity names and public figures in text. "
            "Extract ONLY the names of real celebrities, politicians, public figures, or fictional characters "
            "mentioned in the text. Return each name on a new line. "
            "Include ONLY the names, with no additional text, numbers, or punctuation."
        )

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.1
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

def process_segments_with_context(segments, celebrities):
    """Process segments while maintaining context across multiple segments."""
    if not segments or not celebrities:
        return [{"character": "Narrator", "text": segment} for segment in segments]
    
    try:
        # Prepare larger context windows (up to 5 segments at a time)
        context_windows = []
        for i in range(len(segments)):
            start = max(0, i-2)
            end = min(len(segments), i+3)
            context = " ".join(segments[start:end])
            context_windows.append({
                "segment_index": i,
                "context": context,
                "segment": segments[i]
            })
        
        # Process in batches of 10 context windows
        batch_size = 10
        all_results = []
        
        for i in range(0, len(context_windows), batch_size):
            batch = context_windows[i:min(i+batch_size, len(context_windows))]
            batch_results = analyze_batch_with_context(batch, celebrities)
            all_results.extend(batch_results)
        
        # Sort results by original segment index
        all_results.sort(key=lambda x: x["segment_index"])
        
        # Post-processing for consistency
        return post_process_for_consistency([{
            "character": item["character"],
            "text": item["segment"]
        } for item in all_results])
    
    except Exception as e:
        logging.error(f"Error processing segments with context: {traceback.format_exc()}")
        return [{"character": "Narrator", "text": segment} for segment in segments]

def analyze_batch_with_context(batch_contexts, celebrities):
    """Analyze a batch of segments with their contexts to determine the subject celebrity."""
    try:
        # Prepare the prompt with all contexts and segments
        context_texts = []
        for i, item in enumerate(batch_contexts):
            context_texts.append(f"CONTEXT {i+1}: {item['context']}\nSEGMENT {i+1} TO ANALYZE: {item['segment']}")
        
        context_text = "\n\n".join(context_texts)
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
        )
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CELEBRITIES: {celebrity_list}\n\n{context_text}"}
            ],
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip().split("\n")
        results = []
        
        # Parse the response to extract character assignments
        for line in response_text:
            if not line or "SEGMENT" not in line or ":" not in line:
                continue
                
            try:
                # Extract segment number and character
                segment_part, character = line.split(":", 1)
                segment_num = int(re.search(r'SEGMENT\s+(\d+)', segment_part).group(1))
                
                if segment_num <= len(batch_contexts):
                    character = character.strip()
                    
                    # Verify the character is in our list or is "Narrator"
                    if character == "Narrator" or character in celebrities:
                        results.append({
                            "segment_index": batch_contexts[segment_num-1]["segment_index"],
                            "segment": batch_contexts[segment_num-1]["segment"],
                            "character": character
                        })
                    else:
                        # Try to find the closest match in the celebrities list
                        best_match = difflib.get_close_matches(character, celebrities, n=1, cutoff=0.8)
                        if best_match:
                            results.append({
                                "segment_index": batch_contexts[segment_num-1]["segment_index"],
                                "segment": batch_contexts[segment_num-1]["segment"],
                                "character": best_match[0]
                            })
                        else:
                            results.append({
                                "segment_index": batch_contexts[segment_num-1]["segment_index"],
                                "segment": batch_contexts[segment_num-1]["segment"],
                                "character": "Narrator"
                            })
            except Exception as e:
                logging.error(f"Error parsing line '{line}': {e}")
        
        # Ensure all segments from the batch are in the results
        for item in batch_contexts:
            if not any(r["segment_index"] == item["segment_index"] for r in results):
                results.append({
                    "segment_index": item["segment_index"],
                    "segment": item["segment"],
                    "character": "Narrator"  # Default fallback
                })
        
        return results
    
    except Exception as e:
        logging.error(f"Error analyzing batch with context: {traceback.format_exc()}")
        return [{
            "segment_index": item["segment_index"],
            "segment": item["segment"],
            "character": "Narrator"
        } for item in batch_contexts]

def post_process_for_consistency(assigned_segments):
    """Apply additional rules to ensure consistency in character assignments."""
    if len(assigned_segments) <= 1:
        return assigned_segments
    
    processed = assigned_segments.copy()
    
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

def analyze_celebrity_content(narration_text):
    """
    Analyzes text to identify mentioned celebrities and segment the text with context awareness.
    Returns segments with identified celebrities.
    """
    if not narration_text or not isinstance(narration_text, str):
        raise ValueError("Invalid narration text provided")
    
    try:
        # First extract all celebrities mentioned in the entire text
        all_celebrities = extract_celebrity_names(narration_text)
        logging.info(f"Extracted celebrities: {all_celebrities}")
        
        if not all_celebrities:
            return [{"character": "Narrator", "text": narration_text}]
        
        # Split text into manageable segments
        segments = estimate_segments(narration_text)
        if not segments:
            return [{"character": "Narrator", "text": narration_text}]
        
        # Process segments with context awareness
        return process_segments_with_context(segments, all_celebrities)
        
    except Exception as e:
        logging.error(f"Error analyzing celebrity content: {traceback.format_exc()}")
        return [{"character": "Narrator", "text": narration_text}]

def get_main_characters(segments, min_mentions=2):
    """Identify main characters based on frequency of mentions."""
    character_counts = Counter([segment["character"] for segment in segments 
                               if segment["character"] != "Narrator"])
    
    # Characters with at least min_mentions are considered main characters
    main_characters = [char for char, count in character_counts.items() 
                      if count >= min_mentions]
    
    return main_characters

def segment_narration(narration_text, characters=None):
    """
    Main function to segment narration and assign characters with context awareness.
    If characters are not provided, it will extract them from the text.
    """
    try:
        # If characters not provided, analyze the text to extract them
        if not characters:
            # First analyze to get segments with potential characters
            initial_segments = analyze_celebrity_content(narration_text)
            
            # Then determine main characters based on frequency
            characters = get_main_characters(initial_segments)
            logging.info(f"Main characters identified: {characters}")
            
            # If we found main characters, re-analyze with just those
            if characters:
                segments = estimate_segments(narration_text)
                return process_segments_with_context(segments, characters)
            else:
                return initial_segments
        else:
            # If characters are provided, use them directly
            segments = estimate_segments(narration_text)
            return process_segments_with_context(segments, characters)
    
    except Exception as e:
        logging.error(f"Error in segment_narration: {traceback.format_exc()}")
        return [{"character": "Narrator", "text": narration_text}]

# Example usage

# narration_text = """
# Hey gossip fans, and welcome back to our channel! Today, we're diving into the hilariously turbulent world of Saturday Night Live, and guess what? It involves some of the biggest celebs that you might've totally forgotten once graced Studio 8H. But first, let's talk about a legendary meltdown that could have cost a career, but instead, became a footnote in the saga of one of comedy’s most iconic figures.

# Yes, I’m talking about none other than Larry David. Picture this: It's the mid-1980s, the air is thick with hairspray and comedy sketches, and Larry David is a writer for SNL. Now, Larry, as you might know from his later work on "Curb Your Enthusiasm," has a bit of a... let’s say, unique personality. He’s brilliant but not exactly the poster boy for patience, and SNL was putting that to the test.

# Week after week, Larry’s sketches smash it at the readthroughs, everyone’s laughing, the atmosphere's electric, and then, bam! Cut. His sketches are axed from the live show repeatedly. Can you imagine the frustration building up? It’s like baking the world's best chocolate cake, and then, just as you're about to serve it, someone tosses it straight into the trash - every single week!

# Now let's fast forward to one infamous night. It's nearly showtime, mere minutes left, and another Larry David original gets the chop. This time, he snaps. Right there in the chaos of pre-live madness, he storms up to the executive producer, Dick Ebersol, and unleashes a torrent of fury. "This f--king show stinks! It’s s--t! I’m done! I’m gone! F--k this! I’m out!” Yes, he actually said that! And off he went, striding out into the frigid New York night, probably fuming and steaming more than the manholes he was stepping over.

# But here comes the twist—by the next morning, with his anger cooled off and the reality of a missing paycheck settling in, Larry does something unimaginably bold. He strolls back into SNL the following week as if his explosive exit was just a figment of everyone’s imagination. Imagine walking into the writers' meeting, all eyes on him, whispers all around, and he just casually pitches a new sketch about a circus. 

# And they let him stay! He only lasted about a year after that, but during his time there, he met Julia Louis-Dreyfus, setting the stage for future collaborations that we all know and love.

# So, not only did Larry David manage to keep his job after declaring the show absolute garbage, but he went on to become a legendary figure in comedy. Stick around as we uncover more shocking stories of celebs and their SNL escapades that you’ve probably forgotten. You won't believe who else has walked through those studio doors!

# But that's just one slice of the saga. The SNL vault is deep with juicy tidbits and forgotten stars who once walked its notorious halls. Let's crank up the way-back machine because next up is a true Hollywood A-lister whose SNL stint might just have you saying, "Wait, what?"

# Enter Robert Downey Jr. Yes, the Iron Man himself! It's hard to imagine RDJ, the suave, quip-dropping superhero, as a cast member on SNL. But cast your mind back to the 1984-1985 season. That's right, before the days of Marvel dominance and blockbuster spotlights, Robert was a fresh-faced, 20-year-old sketch comedy actor on SNL.

# Now picture this young, eager RDJ, thrown into the competitive ring that is SNL. It was a season known for its dramatic overhaul, with a completely new cast plucked straight from the fringes of fame. But despite the electricity of new beginnings, it wasn't all smooth sailing. The season itself faced harsh criticism; it was deemed one of the less successful in the show's run, and poor RDJ, well, let’s just say not many of his characters are etched in the annals of comedy glory.

# Imagine the frustration of a young artist, ready to make his mark, only to find that the platform just isn't fitting right. Week after week, sketches fell flat, audiences were tough, and the spotlight seemed just out of reach. It's a year of potential that never fully materialized on the SNL stage. And yet, for RDJ, it was but a brief stop on his meteoric rise to stardom.

# Moving on from Iron Man to a woman of steel, did you know that Julia Louis-Dreyfus—yes, Elaine from "Seinfeld" and Selina Meyer from "Veep"—was also a bright-eyed player on the SNL stage? Joining in the early 1980s, like RDJ and Larry David, she found herself amidst the chaos and charm of the SNL family. Rumor has it her time at SNL was packed with potential but hampered by less-than-stellar scripts and a show in a transition period.

# Julia, the comedic genius who would later snag countless Emmys, once walked through Studio 8H, likely pondering sketch ideas and dodging those notorious last-minute cuts. Can you feel her exasperation? There she was, a future queen of comedy, jesting and jiving in bits that often didn’t make it past dress rehearsal.

# Fast forward and add into the mix, none other than Joan Cusack. That's right, before she was handling the shenanigans of the Gallagher family in "Shameless," Joan was trying to find her comedic timing on SNL during that same tumultuous 1984-1985 season as RDJ and Julia. It's said her time on SNL was challenging, facing stiff competition from established comedies of the time, like SCTV. The frustration of creating—crafting characters, hoping for laughs, all under the pressure cooker of live television—is something Joan faced head-on in Studio 8H.

# Each of these stars—RDJ, Julia, Joan, and yes, our man Larry David—shared that same space, breathed that hallowed SNL air, and faced their own battles with the beast of live sketch comedy. Some nights felt like a knock-out punch of disappointment as sketches were axed and characters flopped, while other nights promised glimmers of the brilliance for which they'd later be universally celebrated.

# So next time you see RDJ zipping through the sky in a high-tech suit, or Julia Louis-Dreyfus ruling the Veep's office with a razor-sharp wit, or even Joan Cusack managing a raucous household on-screen, remember where they once stood. They stood where every iconic SNL cast member has stood—under the bright lights, in front of the live cameras, facing down the exhilarating, maddening, and utterly unique challenge that is Saturday Night Live.

# And what about Larry? Well, despite that explosive, near-career-ending frustration, his story reminds us all that sometimes, the path to legendary status is fraught with moments of doubt, battles with the establishment, and, yes, sheer comedic chaos.

# Instead of fading away, these stars used their frustrations as fuel, propelling them onto paths that would one day make them household names. And isn't that the ultimate SNL silver lining? The frustration, the fight, the raw unfiltered challenge of it all—it didn't define them. It merely forged them in the fire of live television, readying them for the great stages yet to come.

# Stay tuned, and keep laughing, because in the world of comedy, you never know which frustrated newcomer today might be tomorrow’s legendary star.

# As we've peeled back the layers of SNL's star-studded history, it's clear that the spotlight has illuminated some surprising faces over the years. From Robert Downey Jr.'s early comic attempts to Larry David’s explosive walkout, which, frankly, might have been his funniest sketch ever. These tales are not just footnotes in their careers but pivotal moments that shaped these icons in ways we're only beginning to understand. The question remains: what other secrets does the stage of SNL hold? Who else walked through those doors, only to become the mega-stars of today?

# If you loved diving into these forgotten chapters of SNL history and crave more juicy revelations, don’t forget to smash that like button. Dive into the comments and let us know which SNL alum surprised you the most, and who you want us to uncover next. Be sure to subscribe and ring the bell for more gossip that you didn’t know you needed, right here, where the past meets the scandalous present! Keep guessing, keep laughing, and until next time, keep tuning in!
# """

# # characters = ["Trump", "Chrissy Teigen", "Whoopi Goldberg","Angela Rye","John Oliver","George Clooney","Meryl Streep","Leonardo DiCaprio","Lizzo","Taylor Swift"]

# segments = segment_narration(narration_text)

# print(segments)

# def time_segments(audio_path, segments):
#     """Estimate time boundaries for each segment based on word count proportionality."""
#     try:
#         if not isinstance(audio_path, str):
#             raise ValueError(f"audio_path must be a valid file path (str), but received: {type(audio_path)}")
#         if not isinstance(segments, list) or not all(isinstance(seg, str) for seg in segments):
#             raise ValueError("Segments must be a list of strings.")

#         # Get total duration of the audio file
#         with AudioFileClip(audio_path) as audio:
#             total_duration = audio.duration

#         # Compute segment weights based on word count
#         segment_word_counts = [len(seg.split()) for seg in segments]
#         total_words = sum(segment_word_counts)

#         if total_words == 0:
#             raise ValueError("All segments are empty; cannot allocate time.")

#         # Calculate proportional time allocation
#         time_allocations = [(word_count / total_words) * total_duration for word_count in segment_word_counts]

#         # Generate time boundaries
#         time_boundaries = []
#         start_time = 0.0

#         for duration in time_allocations:
#             end_time = start_time + duration
#             time_boundaries.append((round(start_time, 2), round(end_time, 2)))
#             start_time = end_time  # Move to next segment start time

#         return time_boundaries

#     except Exception as e:
#         raise RuntimeError(f"Error in time_segments function: {str(e)}")

# for segment in segments:
#     print(f"{segment['character']}: {segment['text']}")
# voice_over_path = "narration_20250219_171843.mp3" 
# time_seg=time_segments(voice_over_path,segments)
# print(time_seg)


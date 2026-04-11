import os
import asyncio
import openai
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Load API credentials
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

async def generate_intro(news_description, emotion, duration,pbar, optionalPrompt):
    """Generate an engaging and drama-filled intro for a celebrity gossip YouTube video."""

    prompt = f"""
    **Your Task:**  
    Write an **irresistible intro** (30-45, 100-150 words) for a YouTube **celebrity gossip and drama** video based on the following news strictly following Youtube's Guidlines:

    **description:** {news_description}  

    **🔥 What This Intro Needs to Achieve:**  
    - Hook the audience **immediately** with **suspense, curiosity, or drama**. 
    - **{optionalPrompt}**
    - Make it feel like **insider gossip**—like spilling tea to a close friend.  
    - Set the stage for **scandal, controversy, or shocking twists**. 
    - **Focus on the celebrities involved**—their reactions, drama, and hidden details.  
    - Focus on this emotion {emotion}, but make it sound **natural and juicy** .  

    **🛑 Important Rules:**  
    - **DO NOT** introduce yourself—just jump straight into the drama.  
    - **{optionalPrompt}**
    - **DO NOT** use section labels (like "Intro:").  
    - **DO NOT** over-explain; let the drama unfold naturally.  
    - Write in a **conversational, fast-paced, and gossipy tone**—like spilling hot tea!  
    - **No stage directions**, just a script that feels **natural when read aloud**.  

    **Example Hooks:**  
    - "Oh boy, this just got messy! You won’t believe what [celebrity] just did— and let’s just say, the internet is losing its mind over it."  
    - "Yikes! Imagine having the entire world dragging you over one mistake... well, that’s exactly what’s happening to [celebrity] right now!"  
    - "Okay, let’s break this down because the drama is **too good** to ignore. It all started when [celebrity] said [controversial statement]... and um, things **spiraled FAST.**"  

    **Now, go all in—give me a hook that’ll make people stay glued to the screen!**
    """

    response = await openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are a YouTube script expert specializing in celebrity gossip and drama content."},
                  {"role": "user", "content": prompt}]
    )

    pbar.update(1)  # Update progress bar
    return response.choices[0].message.content.strip()

async def generate_body(news_description, intro_text, emotion, duration,pbar,optionalPrompt):
    """Generate the dramatic and gossipy body section of a celebrity news YouTube video with optimized clip placement."""

    prompt = f"""
    **Your Task:**  
    Continue the **celebrity gossip and drama YouTube script body** ({duration} minutes, {duration*150}-{duration*200} words) based on the following details:

    **Description:** {news_description}  

    **🌟 Previously Generated Intro:**  
    {intro_text}  
    **🔥 What This Section Needs to Achieve:**
    - Lenght of text MUST be {duration*150}-{duration*200} words including a conclusion.
    - Focus on this emotion {emotion}, but make it sound **natural and juicy** 
    - **{optionalPrompt}**
    - Keep the story **flowing smoothly** like a juicy, well-structured gossip.  
    - Maintain a **gossipy, dramatic, and highly engaging** tone.  
    - Focus on **scandal, controversy, feuds, or shocking twists**.  
    - Emphasize **the emotions, reactions, and hidden drama** of the celebrities involved.  
    - Present key points **in a chronological order** to make it **easy to follow**.  
    - Use **exaggeration and suspense** to keep the viewer hooked!  
    - **Identify where video clips should be inserted** by using:  
      - **[CLIP: Exact moment & description of clip]**  
      - Example: **[CLIP: News anchor reacting to the scandal]**  
      - Example: **[CLIP: Celebrity’s Instagram response]**  
      - Example: **[CLIP: Fan outrage on Twitter]**  

    **🚫 Strict Rules:**  
    - **USE [CLIP: description of kind of clip] in the most effective places.**  
    - **NO SECTION LABELS** (like "Body:").  
    - **NO stage directions**—just a natural, fast-paced narration.  
    - **NO generic commentary**—this needs to feel like spilling hot celebrity tea!  
    - **Use conversational language**—like you’re telling this to a friend.
    - - **{optionalPrompt}** 
    - If there’s a feud, **highlight both sides** and the public reaction.  

    **Example Phrases for Drama:**  
    - "Oof, can you imagine? [Celebrity] must be **screaming** right now!"  
    - "And here’s where things get *even messier*—because guess what? [Celebrity] just clapped back... and it’s SAVAGE!"  
    - "The internet wasted **zero time** dragging [celebrity] for this, and let’s just say... it’s **not** looking good for them."  

    🚨 **Important:**  
    👉 **Ensure that video clip placements align with key moments for maximum impact!**  
    👉 **Clip placement should add suspense, visual proof, or reaction to intensify the drama.**  

    **Now, let’s dive into the full drama—make it juicy!**
    """


    response = await openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a YouTube script expert specializing in celebrity gossip and drama content."},
                {"role": "user", "content": prompt}
            ]
        )

    pbar.update(1)  # Update progress bar
    return response.choices[0].message.content.strip()
   

# async def generate_conclusion(news_description, body_text, emotion, pbar):
#     """Generate the dramatic conclusion of a celebrity gossip YouTube video."""

#     # Use only part of the body for context
#     summary_context = "..." if len(body_text) < 500 else body_text[:500] + "..."

#     prompt = f"""
#     **Your Task:**  
#     Write a **spicy, dramatic, and engaging conclusion** (30 seconds, 70-150 words) for a **celebrity gossip and drama** YouTube video.

#     **Description:** {news_description}  

#     **🔥 Context from the Story:**  
#     {summary_context}  

#     **🎭 What This Conclusion Needs to Achieve:**  
#     - Wrap up the drama with a **strong closing statement or a cliffhanger**.  
#     - Leave the audience **thinking about what happens next**.  
#     - Use **suspense, mystery, or an open-ended question** to drive engagement.  
#     - End with a **natural call to action** that encourages comments, likes, and subscriptions.  

#     **🚫 Strict Rules:**  
#     - **NO section labels** (like "Conclusion:").  
#     - **NO generic summary**—this needs to **end with impact!**  
#     - **NO robotic or scripted CTA**—it should feel natural, like a real conversation.  

#     **Example Endings:**  
#     - "So... do you think [celebrity] was **totally out of line**, or do they have a point? Let’s chat in the comments!"  
#     - "Whew! This is **far from over**, and you **already know** we’ll be here to spill the updates. So hit that subscribe button, and I’ll catch you in the next one!"  
#     - "Honestly? This could change everything for [celebrity]. But hey—what do you think? Drop your take in the comments!"  

#     **Now, wrap this up with a bang!**
#     """

#     response = await openai_client.chat.completions.create(
#         model="gpt-4-turbo",
#         messages=[{"role": "system", "content": "You are a YouTube script expert specializing in celebrity gossip and drama content."},
#                   {"role": "user", "content": prompt}]
#     )

#     pbar.update(1)  # Update progress bar
#     return response.choices[0].message.content.strip()

async def generate_viral_title(script_content):
    """Generate YouTube title from script content"""
    prompt = f"""
    Create a viral YouTube title based on this script:
    {script_content[:2000]}... [truncated]
    
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
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()

async def generate_full_script_async(news_description, emotion,duration, optionalPrompt):
    """Generate the full script asynchronously with a progress loader."""
    with tqdm(total=3, desc="Generating Script", unit="section") as pbar:
        intro = await generate_intro(news_description, emotion,duration,pbar, optionalPrompt)
        body = await generate_body(news_description, intro, emotion,duration, pbar, optionalPrompt)
        title=await generate_viral_title(intro)
        # conclusion = await generate_conclusion(news_description, body, emotion, pbar)

    full_script = f"{intro}\n\n{body}"
    print("\nScripts Generated ✅")
    
    return {"intro": intro, "body": body, "fullScript": full_script,"title":title}

def generate_full_script( news_description, emotion, duration,optionalPrompt):
    """Runs the async script generator in a synchronous environment."""
    return asyncio.run(generate_full_script_async(news_description, emotion, duration,optionalPrompt))


# import os
# import asyncio
# from mistralai import Mistral
# from dotenv import load_dotenv
# from tqdm.asyncio import tqdm
# import logging
# import re
# import difflib
# import traceback
# from collections import Counter

# # Load API credentials
# load_dotenv()
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# # Initialize Mistral AI client
# mistral_client = Mistral(
#         api_key=os.getenv("MISTRAL_API_KEY", ""),
#     )

# async def generate_script_and_extract_data(title, description, emotion, duration, optional_prompt):
#     """Generate the full script and extract main characters, keywords, and visual summaries."""

#     async def generate_intro(news_title, news_description, emotion, optional_prompt):
#         """Generate an engaging and drama-filled intro for a celebrity gossip YouTube video."""
#         prompt = f"""
#         **Your Task:**
#         Write an **irresistible intro** (30-45, 100-150 words) for a YouTube **celebrity gossip and drama** video based on the following news strictly following Youtube's Guidelines:

#         **Title:** {news_title}
#         **Description:** {news_description}

#         **🔥 What This Intro Needs to Achieve:**
#         - Hook the audience **immediately** with **suspense, curiosity, or drama**.
#         - Make it feel like **insider gossip**—like spilling tea to a close friend.
#         - Set the stage for **scandal, controversy, or shocking twists**.
#         - **Focus on the celebrities involved**—their reactions, drama, and hidden details.
#         - Lean into the emotion **{emotion}**, but make it sound **natural and juicy**.

#         **🛑 Important Rules:**
#         - **{optional_prompt}.
#         - **DO NOT** introduce yourself—just jump straight into the drama.
#         - **DO NOT** use section labels (like "Intro:").
#         - **DO NOT** over-explain; let the drama unfold naturally.
#         - Write in a **conversational, fast-paced, and gossipy tone**—like spilling hot tea!
#         - **No stage directions**, just a script that feels **natural when read aloud**.

#         **Example Hooks:**
#         - "Oh boy, this just got messy! You won’t believe what [celebrity] just did— and let’s just say, the internet is losing its mind over it."
#         - "Yikes! Imagine having the entire world dragging you over one mistake... well, that’s exactly what’s happening to [celebrity] right now!"
#         - "Okay, let’s break this down because the drama is **too good** to ignore. It all started when [celebrity] said [controversial statement]... and um, things **spiraled FAST.**"

#         **Now, go all in—give me a hook that’ll make people stay glued to the screen!**
#         """

#         response = await mistral_client.chat.completions.create(
#             model="mistral-small-latest",  # Use the appropriate model name
#             messages=[{"role": "system", "content": "You are a YouTube script expert specializing in celebrity gossip and drama content."},
#                       {"role": "user", "content": prompt}],
#             max_tokens=150,
#             temperature=0.7,
#             top_p=1,
#             n=1
#         )

#         return response['choices'][0]['message']['content'].strip()

#     async def generate_body(news_title, news_description, intro_text, emotion, duration, optional_prompt):
#         """Generate the dramatic and gossipy body section of a celebrity news YouTube video."""
#         prompt = f"""
#         **Your Task:**
#         Continue the **celebrity gossip and drama YouTube script body** ({duration} minutes, {duration*150}-{duration*200} words) based on the following details:

#         **Title:** {news_title}
#         **Description:** {news_description}

#         **🌟 Previously Generated Intro:**
#         {intro_text}

#         **🔥 What This Section Needs to Achieve:**
#         - Keep the story **flowing smoothly** like a juicy, well-structured gossip.
#         - Maintain a **gossipy, dramatic, and highly engaging** tone.
#         - Focus on **scandal, controversy, feuds, or shocking twists**.
#         - Emphasize **the emotions, reactions, and hidden drama** of the celebrities involved.
#         - Present key points **in a chronological order** to make it **easy to follow**.
#          - Lean into the emotion **{emotion}**, but make it sound **natural and juicy**.
#         - Use **exaggeration and suspense** to keep the viewer hooked!

#         **🚫 Strict Rules:**
#         - **{optional_prompt}
#         - **NO SECTION LABELS** (like "Body:").
#         - **NO stage directions**—just a natural, fast-paced narration.
#         - **NO generic commentary**—this needs to feel like spilling hot celebrity tea!
#         - **Use conversational language**—like you’re telling this to a friend.
#         - If there’s a feud, **highlight both sides** and the public reaction.

#         **Example Phrases for Drama:**
#         - "Oof, can you imagine? [Celebrity] must be **screaming** right now!"
#         - "And here’s where things get *even messier*—because guess what? [Celebrity] just clapped back... and it’s SAVAGE!"
#         - "The internet wasted **zero time** dragging [celebrity] for this, and let’s just say... it’s **not** looking good for them."

#         **Now, let’s dive into the full drama—make it juicy!**
#         """

#         response = await mistral_client.chat.completions.create(
#             model="mistral-small-latest",
#             messages=[{"role": "system", "content": "You are a YouTube script expert specializing in celebrity gossip and drama content."},
#                       {"role": "user", "content": prompt}],
#             max_tokens=duration*200,
#             temperature=0.7,
#             top_p=1,
#             n=1
#         )

#         return response['choices'][0]['message']['content'].strip()

#     async def generate_conclusion(news_title, news_description, body_text, emotion):
#         """Generate the dramatic conclusion of a celebrity gossip YouTube video."""
#         summary_context = "..." if len(body_text) < 500 else body_text[:500] + "..."

#         prompt = f"""
#         **Your Task:**
#         Write a **spicy, dramatic, and engaging conclusion** (30 seconds, 70-150 words) for a **celebrity gossip and drama** YouTube video.

#         **Title:** {news_title}
#         **Description:** {news_description}

#         **🔥 Context from the Story:**
#         {summary_context}

#         **🎭 What This Conclusion Needs to Achieve:**
#         - Wrap up the drama with a **strong closing statement or a cliffhanger**.
#         - Leave the audience **thinking about what happens next**.
#         - Use **suspense, mystery, or an open-ended question** to drive engagement.
#         - End with a **natural call to action** that encourages comments, likes, and subscriptions.

#         **🚫 Strict Rules:**
#         - **NO section labels** (like "Conclusion:").
#         - **NO generic summary**—this needs to **end with impact!**
#         - **NO robotic or scripted CTA**—it should feel natural, like a real conversation.

#         **Example Endings:**
#         - "So... do you think [celebrity] was **totally out of line**, or do they have a point? Let’s chat in the comments!"
#         - "Whew! This is **far from over**, and you **already know** we’ll be here to spill the updates. So hit that subscribe button, and I’ll catch you in the next one!"
#         - "Honestly? This could change everything for [celebrity]. But hey—what do you think? Drop your take in the comments!"

#         **Now, wrap this up with a bang!**
#         """

#         response = await mistral_client.chat.completions.create(
#             model="mistral-small-latest",
#             messages=[{"role": "system", "content": "You are a YouTube script expert specializing in celebrity gossip and drama content."},
#                       {"role": "user", "content": prompt}],
#             max_tokens=150,
#             temperature=0.7,
#             top_p=1,
#             n=1
#         )

#         return response['choices'][0]['message']['content'].strip()

#     async def extract_celebrity_names(text):
#         """Extract potential celebrity names from text using NER via the API."""
#         try:
#             system_prompt = (
#                 "You are an expert in identifying celebrity names and public figures in text. "
#                 "Extract ONLY the names of real celebrities, politicians, public figures, or fictional characters "
#                 "mentioned in the text. Return each name on a new line. "
#                 "Include ONLY the names, with no additional text, numbers, or punctuation."
#             )

#             response = await mistral_client.chat.completions.create(
#                 model="mistral-small-latest",
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": text}
#                 ],
#                 max_tokens=100,
#                 temperature=0.7,
#                 top_p=1,
#                 n=1
#             )

#             extracted_names = response['choices'][0]['message']['content'].strip().split('\n')
#             validated_names = []
#             for name in extracted_names:
#                 name = name.strip()
#                 if name and re.match(r'^[A-Za-z\s\'-]+$', name):
#                     validated_names.append(name)

#             return validated_names

#         except Exception as e:
#             logging.error(f"Error extracting celebrity names: {e}")
#             return []

#     async def process_segments_with_context(segments, celebrities):
#         """Process segments while maintaining context across multiple segments."""
#         if not segments or not celebrities:
#             return [{"character": "Narrator", "text": segment} for segment in segments]

#         try:
#             context_windows = []
#             for i in range(len(segments)):
#                 start = max(0, i-2)
#                 end = min(len(segments), i+3)
#                 context = " ".join(segments[start:end])
#                 context_windows.append({
#                     "segment_index": i,
#                     "context": context,
#                     "segment": segments[i]
#                 })

#             batch_size = 10
#             all_results = []

#             for i in range(0, len(context_windows), batch_size):
#                 batch = context_windows[i:min(i+batch_size, len(context_windows))]
#                 batch_results = await analyze_batch_with_context(batch, celebrities)
#                 all_results.extend(batch_results)

#             all_results.sort(key=lambda x: x["segment_index"])
#             return post_process_for_consistency([{
#                 "character": item["character"],
#                 "text": item["segment"],
#                 "description": item.get("description", "")
#             } for item in all_results])

#         except Exception as e:
#             logging.error(f"Error processing segments with context: {traceback.format_exc()}")
#             return [{"character": "Narrator", "text": segment} for segment in segments]

#     async def analyze_batch_with_context(batch_contexts, celebrities):
#         """Analyze a batch of segments with their contexts to determine the subject celebrity."""
#         try:
#             context_texts = []
#             for i, item in enumerate(batch_contexts):
#                 context_texts.append(f"CONTEXT {i+1}: {item['context']}\nSEGMENT {i+1} TO ANALYZE: {item['segment']}")

#             context_text = "\n\n".join(context_texts)
#             celebrity_list = ", ".join(celebrities)

#             system_prompt = (
#                 f"You are an expert in analyzing text and determining who is being discussed.\n\n"
#                 f"For each numbered segment, determine which person from this list is being discussed: {celebrity_list}.\n\n"
#                 f"IMPORTANT RULES:\n"
#                 f"1. The segment may not explicitly mention a person by name, but may refer to them via context.\n"
#                 f"2. If a segment continues discussing the same person from previous segments, assign that person.\n"
#                 f"3. If the segment discusses someone's tweet, quote, action, or opinion, assign that person.\n"
#                 f"4. If the segment contains a direct quote, assign the person who said it.\n"
#                 f"5. Assign 'Narrator' ONLY if the segment truly doesn't discuss any specific person.\n"
#                 f"6. DO NOT default to 'Narrator' - carefully analyze context to identify who is being discussed.\n"
#                 f"7. Return ONLY the format 'SEGMENT X: PersonName' for each segment."
#             )

#             response = await mistral_client.chat.completions.create(
#                 model="mistral-small-latest",
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": f"CELEBRITIES: {celebrity_list}\n\n{context_text}"}
#                 ],
#                 max_tokens=100,
#                 temperature=0.7,
#                 top_p=1,
#                 n=1
#             )

#             response_text = response['choices'][0]['message']['content'].strip().split("\n")
#             results = []

#             for line in response_text:
#                 if not line or "SEGMENT" not in line or ":" not in line:
#                     continue

#                 try:
#                     segment_part, character = line.split(":", 1)
#                     segment_num = int(re.search(r'SEGMENT\s+(\d+)', segment_part).group(1))

#                     if segment_num <= len(batch_contexts):
#                         character = character.strip()

#                         if character == "Narrator" or character in celebrities:
#                             results.append({
#                                 "segment_index": batch_contexts[segment_num-1]["segment_index"],
#                                 "segment": batch_contexts[segment_num-1]["segment"],
#                                 "character": character
#                             })
#                         else:
#                             best_match = difflib.get_close_matches(character, celebrities, n=1, cutoff=0.8)
#                             if best_match:
#                                 results.append({
#                                     "segment_index": batch_contexts[segment_num-1]["segment_index"],
#                                     "segment": batch_contexts[segment_num-1]["segment"],
#                                     "character": best_match[0]
#                                 })
#                             else:
#                                 results.append({
#                                     "segment_index": batch_contexts[segment_num-1]["segment_index"],
#                                     "segment": batch_contexts[segment_num-1]["segment"],
#                                     "character": "Narrator"
#                                 })
#                 except Exception as e:
#                     logging.error(f"Error parsing line '{line}': {e}")

#             for item in batch_contexts:
#                 if not any(r["segment_index"] == item["segment_index"] for r in results):
#                     results.append({
#                         "segment_index": item["segment_index"],
#                         "segment": item["segment"],
#                         "character": "Narrator"
#                     })

#             return results

#         except Exception as e:
#             logging.error(f"Error analyzing batch with context: {traceback.format_exc()}")
#             return [{
#                 "segment_index": item["segment_index"],
#                 "segment": item["segment"],
#                 "character": "Narrator"
#             } for item in batch_contexts]

#     def post_process_for_consistency(assigned_segments):
#         """Apply additional rules to ensure consistency in character assignments."""
#         if len(assigned_segments) <= 1:
#             return assigned_segments

#         processed = assigned_segments.copy()

#         for i in range(1, len(processed)):
#             if ('"' in processed[i]["text"] or "'" in processed[i]["text"]) and processed[i]["character"] == "Narrator":
#                 if processed[i-1]["character"] != "Narrator":
#                     processed[i]["character"] = processed[i-1]["character"]

#         i = 0
#         while i < len(processed):
#             if processed[i]["character"] != "Narrator":
#                 current_character = processed[i]["character"]

#                 j = i + 1
#                 context_ended = False

#                 while j < len(processed) and not context_ended:
#                     if processed[j]["character"] != "Narrator" and processed[j]["character"] != current_character:
#                         context_ended = True

#                     elif processed[j]["character"] == "Narrator":
#                         segment_lower = processed[j]["text"].lower()
#                         pronouns = ["he", "she", "they", "his", "her", "their", "them"]
#                         context_words = ["tweet", "statement", "said", "mentioned", "post", "comment", "response"]

#                         if any(pronoun in segment_lower.split() for pronoun in pronouns) or \
#                            any(word in segment_lower for word in context_words):
#                             processed[j]["character"] = current_character

#                     j += 1

#             i += 1

#         return processed

#     async def analyze_celebrity_content(narration_text):
#         """
#         Analyzes text to identify mentioned celebrities and segment the text with context awareness.
#         Returns segments with identified celebrities.
#         """
#         if not narration_text or not isinstance(narration_text, str):
#             raise ValueError("Invalid narration text provided")

#         try:
#             all_celebrities = await extract_celebrity_names(narration_text)
#             logging.info(f"Extracted celebrities: {all_celebrities}")

#             if not all_celebrities:
#                 return [{"character": "Narrator", "text": narration_text}]

#             segments = estimate_segments(narration_text)
#             if not segments:
#                 return [{"character": "Narrator", "text": narration_text}]

#             return await process_segments_with_context(segments, all_celebrities)

#         except Exception as e:
#             logging.error(f"Error analyzing celebrity content: {traceback.format_exc()}")
#             return [{"character": "Narrator", "text": narration_text}]

#     def get_main_characters(segments, min_mentions=2):
#         """Identify main characters based on frequency of mentions."""
#         character_counts = Counter([segment["character"] for segment in segments
#                                    if segment["character"] != "Narrator"])

#         main_characters = [char for char, count in character_counts.items()
#                           if count >= min_mentions]

#         return main_characters

#     async def segment_narration(narration_text, characters=None):
#         """
#         Main function to segment narration and assign characters with context awareness.
#         If characters are not provided, it will extract them from the text.
#         """
#         try:
#             if not characters:
#                 initial_segments = await analyze_celebrity_content(narration_text)
#                 characters = get_main_characters(initial_segments)
#                 logging.info(f"Main characters identified: {characters}")

#                 if characters:
#                     segments = estimate_segments(narration_text)
#                     return await process_segments_with_context(segments, characters)
#                 else:
#                     return initial_segments
#             else:
#                 segments = estimate_segments(narration_text)
#                 return await process_segments_with_context(segments, characters)

#         except Exception as e:
#             logging.error(f"Error in segment_narration: {traceback.format_exc()}")
#             return [{"character": "Narrator", "text": narration_text}]

#     async def extract_keywords(section_name, content):
#         """Extract main characters (people only) from the given section asynchronously."""
#         if not content.strip():
#             return []

#         prompt = f"""
#         **Task:** Identify and extract the **main people (characters)** mentioned in the following section titled **"{section_name}"**.

#         **Section Content:**
#         {content}

#         **Instructions:**
#         - Extract only **names of people** (characters).
#         - **Do not include** places, organizations, objects, or abstract concepts.
#         - Provide **full names** when possible (e.g., "Elon Musk" instead of just "Elon").
#         - **No duplicates**—each name should appear only once.
#         - If no people are mentioned, return `"None"`.
#         - Output should be a **comma-separated list** of names.

#         **Example Outputs:**
#         - ✅ Input: "Elon Musk and Jeff Bezos had a discussion."
#           - Output: `Elon Musk, Jeff Bezos`
#         - ✅ Input: "This section talks about Tesla and SpaceX."
#           - Output: `"None"`
#         - ✅ Input: "Alexander the Great was a powerful ruler."
#           - Output: `Alexander the Great`
#         """

#         response = await mistral_client.chat.completions.create(
#             model="mistral-small-latest",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=100,
#             temperature=0.7,
#             top_p=1,
#             n=1
#         )

#         keywords = response['choices'][0]['message']['content'].strip()
#         return [] if keywords.lower() == "none" or not keywords else [name.strip() for name in keywords.split(",")]

#     async def extract_visual_summary(section_name, content):
#         """Extract a visual summary from the given section asynchronously."""
#         if not content:
#             return ""

#         prompt = f"""
#         Based on this {section_name}, give a short summary highlighting focus points on characters, images, and clips that will best suit the scenario as per the script.

#         Provide a short but clear **summary**.
#         """

#         response = await mistral_client.chat.completions.create(
#             model="mistral-small-latest",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=100,
#             temperature=0.7,
#             top_p=1,
#             n=1
#         )

#         return response['choices'][0]['message']['content'].strip()

#     async def generate_youtube_keywords_for_characters(text, characters, max_keywords=5):
#         """
#         Generates YouTube search keywords for each character in the list based on the input text.

#         Args:
#             text (str): Input text to analyze
#             characters (list): List of characters/entities to generate keywords for
#             max_keywords (int): Maximum number of keywords to return per character (default: 5)

#         Returns:
#             dict: A dictionary where keys are characters and values are lists of keywords
#         """

#         character_keywords = {}

#         for character in characters:
#             prompt = f"""Analyze the following text and generate {max_keywords} relevant YouTube search keywords
#             specifically related to {character}. Focus on keywords that would be effective for finding related video content.
#             Format the response as a comma-separated list without numbers or bullet points.

#             Text: {text[:3000]}  # Truncate to 3000 chars to avoid token limit

#             Keywords:"""

#             try:
#                 response = await mistral_client.chat.completions.create(
#                     model="mistral-small-latest",
#                     messages=[
#                         {"role": "system", "content": "You are a helpful SEO assistant."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     max_tokens=50,
#                     temperature=0.7,
#                     top_p=1,
#                     n=1
#                 )

#                 if response['choices'] and response['choices'][0]['message']['content']:
#                     raw_output = response['choices'][0]['message']['content'].strip()
#                     keywords = [k.strip() for k in re.split(r',|\n', raw_output) if k.strip()]
#                     character_keywords[character] = keywords[:max_keywords]
#                 else:
#                     character_keywords[character] = []

#             except Exception as e:
#                 print(f"Error generating keywords for {character}: {e}")
#                 character_keywords[character] = []

#         return character_keywords

#     async def extract_main_character_keywords_and_visuals_async(intro, body):
#         """Extracts main characters and visual summaries asynchronously with a progress loader."""

#         with tqdm(total=4, desc="Extracting Data", unit="task") as pbar:
#             intro_segments = await segment_narration(intro)
#             body_segments = await segment_narration(body)
#             pbar.update(1)

#             intro_characters = get_main_characters(intro_segments)
#             body_characters = get_main_characters(body_segments)
#             pbar.update(2)

#             yt_intro_keywords = await generate_youtube_keywords_for_characters(intro, intro_characters)
#             yt_body_keywords = await generate_youtube_keywords_for_characters(body, body_characters)
#             pbar.update(3)

#         extracted_data = {
#             "intro": {
#                 "main_characters": intro_characters,
#                 "yt_keywords": yt_intro_keywords,
#                 "segments": intro_segments
#             },
#             "body": {
#                 "main_characters": body_characters,
#                 "yt_keywords": yt_body_keywords,
#                 "segments": body_segments
#             }
#         }

#         print("\nExtraction Completed ✅")
#         return extracted_data

#     def estimate_segments(narration_text):
#         """Split the narration text into segments, handling various punctuation."""
#         if not narration_text or not isinstance(narration_text, str):
#             return []

#         protected_text = re.sub(r'(\w\.\w\.)', r'\1PROTECTED', narration_text)
#         protected_text = re.sub(r'(Mr\.|Mrs\.|Dr\.|etc\.|i\.e\.|e\.g\.)', r'\1PROTECTED', protected_text)
#         protected_text = re.sub(r'\.\.\.', 'ELLIPSIS', protected_text)

#         raw_segments = re.split(r'(?<!\w\.\w\.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\:)\s', protected_text)

#         segments = []
#         for segment in raw_segments:
#             if segment.strip():
#                 cleaned = segment.replace('PROTECTED', '').replace('ELLIPSIS', '...')
#                 if not cleaned.endswith(('.', '!', '?', ':')):
#                     cleaned += '.'
#                 segments.append(cleaned.strip())

#         return segments

#     async def identify_clip_opportunities(segments):
#         """Identify opportunities for adding video clips in the segments."""
#         clip_opportunities = []
#         for segment in segments:
#             if "said" in segment["text"].lower() or "mentioned" in segment["text"].lower() or "posted" in segment["text"].lower():
#                 clip_opportunities.append(segment["text"])
#         return clip_opportunities

#     async def generate_clip_descriptions(segments, clip_opportunities):
#         """Generate descriptions for video clips based on identified opportunities."""
#         clip_descriptions = []
#         for opportunity in clip_opportunities:
#             prompt = f"""
#             Based on the following segment, generate a short description for a video clip that would provide context or engagement:

#             Segment: {opportunity}

#             Description:
#             """
#             response = await mistral_client.chat.completions.create(
#                 model="mistral-small-latest",
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=50,
#                 temperature=0.7,
#                 top_p=1,
#                 n=1
#             )
#             description = response['choices'][0]['message']['content'].strip()
#             clip_descriptions.append({"segment": opportunity, "description": description})

#         # Merge clip descriptions with segments
#         for clip in clip_descriptions:
#             for segment in segments:
#                 if segment["text"] == clip["segment"]:
#                     segment["description"] = clip["description"]
#                     break

#         return segments

#     # Generate the script
#     intro = await generate_intro(title, description, emotion, optional_prompt)
#     body = await generate_body(title, description, intro, emotion, duration, optional_prompt)
#     conclusion = await generate_conclusion(title, description, body, emotion)

#     full_script = f"{intro}\n\n{body}\n\n{conclusion}"
#     print("\nScripts Generated ✅")

#     # Extract data
#     extracted_data = await extract_main_character_keywords_and_visuals_async(intro, body)

#     # Identify clip opportunities and generate descriptions
#     intro_clip_opportunities = await identify_clip_opportunities(extracted_data["intro"]["segments"])
#     body_clip_opportunities = await identify_clip_opportunities(extracted_data["body"]["segments"])

#     intro_segments_with_clips = await generate_clip_descriptions(extracted_data["intro"]["segments"], intro_clip_opportunities)
#     body_segments_with_clips = await generate_clip_descriptions(extracted_data["body"]["segments"], body_clip_opportunities)

#     # Combine scripts and extracted data
#     result = {
#         "scripts": {
#             "intro": intro,
#             "body": body,
#             "conclusion": conclusion,
#             "fullScript": full_script
#         },
#         "extracted_data": {
#             "intro": {
#                 "main_characters": extracted_data["intro"]["main_characters"],
#                 "yt_keywords": extracted_data["intro"]["yt_keywords"],
#                 "segments": intro_segments_with_clips
#             },
#             "body": {
#                 "main_characters": extracted_data["body"]["main_characters"],
#                 "yt_keywords": extracted_data["body"]["yt_keywords"],
#                 "segments": body_segments_with_clips
#             }
#         }
#     }

#     return result

# def generate_script_and_extract_data_sync(title, description, emotion, duration, optional_prompt):
#     """Runs the async function synchronously in a normal Python environment."""
#     return asyncio.run(generate_script_and_extract_data(title, description, emotion, duration, optional_prompt))

# Example Usage
# news_title = "Elon Musk Shocks The World With His Latest Announcement!"
# news_description = "Elon Musk just dropped a major revelation about Tesla's future, and it's leaving everyone stunned."
# emotion = "shock and excitement"
# duration=7
# optional_prompt=''
# script = generate_script_and_extract_data_sync(news_title, news_description, emotion, duration,optional_prompt)
# print(script)

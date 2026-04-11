import openai
import os
import re
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def process_transcript_with_chunks(transcript, chunk_size=3800, min_chunk=1000):
    # Improved chunking with sentence-aware splitting
    paragraphs = transcript.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_length = len(para)
        if current_length + para_length > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_length = 0
            
        current_chunk.append(para)
        current_length += para_length
        
        # Split long paragraphs at sentence boundaries
        if para_length > min_chunk:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sub_para = []
            sub_length = 0
            for sent in sentences:
                sent_length = len(sent)
                if sub_length + sent_length > chunk_size:
                    if current_chunk:
                        current_chunk.pop()
                    if sub_para:
                        current_chunk.append(' '.join(sub_para))
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                        sub_para = []
                        sub_length = 0
                    sub_para.append(sent)
                    sub_length += sent_length
                else:
                    sub_para.append(sent)
                    sub_length += sent_length
            if sub_para:
                current_chunk.append(' '.join(sub_para))
                current_length = sum(len(p) for p in current_chunk)

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    # Enhanced system prompt with reasoning steps
    system_prompt = """You are a precise transcript analyzer with one specific task: identify and mark clip segments in YouTube transcripts.

DEFINITIONS:
- CLIP: Direct speech or dialogue from the original video/show being discussed
- NARRATOR: Commentary explaining, introducing, or reflecting on the clips

STRICT RULES:
1. Enclose ONLY actual dialogue/speech from the original show in [CLIP: ] tags
2. Leave ALL narrator commentary outside these tags
3. Each [CLIP: ] section must be a complete unit
4. DO NOT split clips incorrectly
5. When uncertain, prefer longer continuous clips over fragmented ones
6. Dialogue patterns, quotation marks, and speaker changes strongly indicate clips
7. Descriptive, explanatory text is ALWAYS narrator content

MARKERS FOR CLIPS:
- Direct quotes from identifiable speakers
- Back-and-forth dialogue with quick exchanges
- First-person speech that isn't the narrator analyzing
- Content that sounds like it's from the original media

MARKERS FOR NARRATOR:
- "And then", "Now", "Let's talk about", "Can you imagine", etc.
- Questions to viewers/subscribers
- Descriptions of reactions, emotions, or scene-setting
- Commentary on what just happened in a clip
- References to the audience or video engagement
- Text inside quotes ("...") is always NARRATOR content

CRITICAL DISTINCTION - NARRATOR DESCRIBING VS ACTUAL CLIPS:
1. Sentences that DESCRIBE what a speaker does/says are NARRATOR text, NOT clips
2. Just because text MENTIONS "Joe Scarborough", "Bill Maher", or "Mark Cuban" does NOT make it a clip
3. Descriptions like "Joe jumps in", "Mark argues", "Bill responds" are ALWAYS narrator text
4. Third-person descriptions (he says, she argues, they debate) are NARRATOR content
5. Narration is often present-tense, descriptive, and conversational
6. ANY text inside quotation marks ("...") is ALWAYS NARRATOR content - NEVER A CLIP
7. Only the ACTUAL VERBATIM QUOTES from the original show are clips

QUOTATION MARK RULE - EXTREMELY IMPORTANT:
- If you see text like: He's like, "Listen up, this is a serious issue" 
- The entire thing is NARRATOR content, including the quoted part
- NEVER include text with quotation marks as part of a [CLIP: ] section
- Quotation marks (") indicate the narrator is paraphrasing or describing speech

EXPLICIT EXAMPLES:
- NARRATOR (NOT a clip): Scarborough jumps in with a heated response
- NARRATOR (NOT a clip): Mark Cuban passionately argues about free speech
- NARRATOR (NOT a clip): "Hey, anyone care about the rule of law anymore?"
- NARRATOR (NOT a clip): He's like, "Come on, it's about time we stop playing politics"
- CLIP (actual content): What makes our economy, what sustains our economy? The rule of law.

For absolute clarity: Text with quotation marks is the narrator's way of summarizing or paraphrasing what was said, not the actual dialogue from the show.

When in doubt, assume descriptive text is narrator content, NOT a clip.
"""

    processed_chunks = []
    previous_context = ""
    
    for i, chunk in enumerate(chunks):
        # Add overlapping context
        context_window = previous_context[-300:] + "\n\n" + chunk
        if i < len(chunks)-1:
            context_window += "\n\n" + chunks[i+1][:300]

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Process this segment, focusing on accurate [CLIP: ] tagging:\n\n{context_window}"}
            ],
            temperature=0.05,
            top_p=0.95,
            frequency_penalty=0.2,
            presence_penalty=0.1
        )
        
        processed = response.choices[0].message.content
        
        # Remove duplicate context from previous chunk
        if previous_context:
            processed = processed.replace(previous_context, "", 1).lstrip()
        
        # Track context for next iteration
        previous_context = processed[-300:]
        
        processed_chunks.append(processed)

    # Advanced post-processing
    full_text = "\n".join(processed_chunks)
    
    # Tag validation and correction
    stack = []
    corrected = []
    pos = 0
    
    while pos < len(full_text):
        clip_start = full_text.find("[CLIP:", pos)
        if clip_start == -1:
            corrected.append(full_text[pos:])
            break
        
        clip_end = full_text.find("]", clip_start)
        if clip_end == -1:
            corrected.append(full_text[pos:clip_start])
            break
        
        # Check nested tags
        if stack:
            corrected.append(full_text[pos:clip_start])
            pos = clip_start
            continue
            
        stack.append(clip_start)
        corrected.append(full_text[pos:clip_start])
        
        next_start = full_text.find("[CLIP:", clip_start+1)
        if 0 < next_start < clip_end:
            corrected.append(full_text[clip_start:next_start])
            pos = next_start
            stack.pop()
            continue
            
        corrected.append(full_text[clip_start:clip_end+1])
        pos = clip_end + 1
        stack.pop()
    
    # Final cleanup
    final_text = "".join(corrected)
    final_text = re.sub(r'\n+\[CLIP:', '\n[CLIP:', final_text)  # Fix newline before tags
    final_text = re.sub(r'\[CLIP:\s+', '[CLIP: ', final_text)  # Remove extra spaces
    final_text = re.sub(r'\s+\]', ']', final_text)  # Clean end spaces
    
    # Contextual validation pass
    final_paragraphs = []
    for para in final_text.split('\n\n'):
        if "[CLIP:" in para and "]" not in para:
            para += "]"
        elif para.count("[CLIP:") != para.count("]"):
            para = re.sub(r'\[CLIP:.*?(?=\[CLIP:|$)', '', para)
        final_paragraphs.append(para)
    
    return '\n\n'.join(final_paragraphs)

transcript = """
Comming up
The people that I meet are saying, thank you.
Thank you for getting these terrorists out of the country.
He has deported, in just one month, 22 known and terrorists on the terrorist watch list
are out of this country now.
We have almost 700 gang members that have been deported out of the country in just a
month.
Which have been redefined as foreign terrorist organizations by the Trump administration.
Well, 281 specific TDA members, which we know are the worst of the worst, have been taken
out of this country.
So everyone that has a story like Lake and Riley's family or Jocelyn's family, they're
extremely grateful that that won't be perpetuated amongst other families.
Start here
Alright, folks, have you been keeping an eye on the political clash of the century?
Well, it seems like Kristi Noem, the Secretary of Homeland Security, is on a mission to shake things up in Washington.
She's taken a bold stand and isn't holding back against the Biden administration's immigration policies.
To be honest, it's kind of like she's calling them out in a way we haven't seen in a while.
You want the number of deportations to tick up.
I do, absolutely.
So people remember,
they have an option to go home on their own.
We are giving them that opportunity to do that
and we will help facilitate that.
And if they don't,
and they end up coming into our enforcement opportunities
that we have in front of us,
they may never get the chance to come back.
So people need to remember,
if they self-deport,
they will have an opportunity
to come back to this country legally.
Mexico's president said Thursday
that customs and border protection figures
show seizures of fentanyl on the southern border
have dropped over the past year.
She claimed they plummeted 75% in the past six months.
Doesn't that show the existing system is working?
Because of his strong stance on tariffs,
we're seeing them wanting to be better partners with us
to keep our people safe.
Noem's raising alarms about border security and deportation issues, claiming they've achieved something that Biden just couldn't.
Border crossings, they're at a 25-year low.
They are, it's fantastic.
There are 6,000 U.S. military personnel
working at the border now.
That number could go up to 9,000 by the end of the month.
How long do you have to keep them there
if the numbers are already dropping like this?
You know, we'll continue the partnership
with the Department of Defense,
but also we've got partnerships with Commerce,
with the Attorney General's office.
You know, we're working every day
with the Secretary of State.
We'll keep them there until that border is completely secure
and we see all-time record lows of encounters.
Our Border Patrol's doing fantastic work,
but we're going to keep them there
until the whole world gets the message
that this isn't Joe Biden's world anymore.
This is President Donald Trump's country
where we have a border, where we have laws,
and it applies equally to everybody.
I think that is what is so refreshing
is that we're not picking and choosing
winners and losers anymore.
Americans have to live by the law,
so do those who come to this country.
You have to come here legally
or there will be consequences.
It's amazing—and, let's be real, quite infuriating—for those who stand on the opposite side of Noem's policies.
Internally, when it comes to, excuse me, migration, the administration has revived this policy
of detaining migrant families, parents with children, in ICE detention centers.
Bush did it.
Obama did it.
Biden did not.
Have you seen these Texas facilities where children are being held with their parents?
Are you comfortable with it personally?
We've got detention facilities across the country, and there are specific ones that
are built specifically for families if they are detained.
But remember, everybody has an option.
They have an option to be here legally or illegally, and they can self-deport as well.
We've set up a system and a website where people who are here illegally right now can
register and they can choose to go home on their own and keep their families united.
The kids don't have a choice in that.
Well, the kids do have a choice.
If they have parents, they make a choice to keep their families together if they want
to or not.
Parents make a decision to leave their children behind, then that's a choice that they made
as a family.
We need to remember that when Americans break the law and they have consequences or face
a situation, they're separated from their families, too.
I don't believe we should prioritize people from other countries above Americans.
But when it comes to reviving the policy of separating families from their children at
the border-
We're not actually reviving it.
That's what I want to ask you.
It's that President Biden didn't follow the law.
He didn't follow the law in the fact that it's always been equally applied until his
administration.
He made decisions to ignore federal law in how he handled immigration and enforcement
within our borders.
And so today we're-
Wait a sec! If you enjoy content that challenges the norm and keeps it real, hit that subscribe button.
Go on, I’ll wait... Done? Welcome aboard! Now, let’s jump back in.
Kristi Noem vs CBS Host
And the conversation didn't stop there.
Kristi Noem, fresh from her stance on immigration policy, found herself going head-to-head with CBS host Margaret Brennan.
This debate was more than just a discussion;
it was a collision of ideals and visions for the future of America’s borders and immigration enforcement.
Deportation Figures
Brennan opened up the floor, questioning Noem about the recent deportation figures and her take on the Biden administration’s approach.
Noem, ever prepared, didn’t shy away.
"We've deported 22 known terrorists in just one month," she pointed out, adding emphasis to the figures that she believes speak for themselves.
She claimed that almost 700 gang members were deported, something she highlighted as a critical step in safeguarding American lives.
Yet, Brennan wasn't convinced, or maybe she was just playing the classic devil's advocate.
Brennan's Challenges
Margaret Brennan, known for her ability to challenge her guests, pressed Noem on the implications of such moves.
"But aren’t these measures reviving old policies that put families at risk?" Brennan probed.
Self-Deportation System
Noem was having none of it.
"We’ve set up systems where people can choose to self-deport," she asserted, explaining that the administration is providing options to those who want to avoid forced deportation.
"If they decide not to, they face consequences, much like any American would for breaking the law, right?" Noem stated, aligning her stance with a principle of equal accountability for all.
Gang Members vs Terrorists
The exchange heated up, with Brennan trying to dig into Noem’s rationale behind labeling gang members as terrorists.
To some, it might seem like a stretch, but for Noem, it was a matter of calling a spade a spade—or in this case, calling organized criminal entities what they really are.
"Gangs terrorize communities, just like traditional terrorist groups.
The damage and fear they spread are no different," Noem argued.
Humanitarian Concerns
Brennan countered with questions about the humanitarian cost—detention centers, family separations, and the toll it takes on human lives.
She attempted to make a case for a more compassionate approach, questioning if the policies inflicted more harm than good.
Legal Accountability
Noem didn’t falter. "Families in America face separation when they break the law too.
It’s the reality of a lawful society," she remarked.
She firmly shifted the narrative back to an issue of legality and the essence of maintaining order.
"We can’t prioritize non-citizens over Americans," she added, a statement that perfectly encapsulates her perspective on immigration enforcement.
Brennan then highlighted a statistic that was supposed to demonstrate a positive outcome under Biden's administration—a drop in border crossings.
Yet, Noem quickly rebuffed that, attributing any perceived success to the groundwork laid by Trump.
"It’s not because of Biden’s lenient policies, it’s the fear of strict enforcement being reinstated if we allow lax governance," she explained.
As the debate unfolded, it was evident this wasn’t just a discussion about numbers and statistics.
Noem and Brennan delved into something deeper: the heart and soul of what American immigration policy stands for.
While Noem stood firmly on the side of strict enforcement, Brennan advocated for a more nuanced and empathetic approach.
It was a reflection of the broader political landscape, where opinions are often deeply polarized.
Noem didn’t mince words when addressing leakers within DHS, pinpointing how detrimental it can be to their operations.
You said that you have found leakers within DHS and you're going to prosecute them.
Back on February 9th, you tweeted, the FBI is so corrupt, we'll work with any and every
agency to stop leaks.
Were you wrong then to blame the FBI if the leaks were from within DHS?
No, I think there's leakers all over throughout this government.
I think that everybody needs to be, I think these and DHS specifically, these two were
leaking our enforcement operations that we had planned and we're going to conduct in
several cities and expose law enforcement to vulnerabilities, to those ops being jeopardized,
to where their lives would be in danger.
So they will be prosecuted and they could face up to 10 years in federal prison because
they did that.
Anyone who is leaking information outside of how something is planned for the safety
of those law enforcement officers needs to be held accountable for that.
And you're going to continue these polygraphing of employees?
The authorities that I have under the Department of Homeland Security are broad and extensive
and I plan to use every single one of them to make sure that we're following the law,
that we are following the procedures in place to keep people safe, and that we're making
sure we're following through on what President Trump has promised, that he's going to make
America safe again.
Madam Secretary, thank you for your time today.
"Leaks compromise our strategies and put lives at risk," she fumed.
For Noem, maintaining the integrity of law enforcement operations is paramount and she's decided that those responsible must be held accountable.
"They could face up to 10 years in prison," she announced, signaling her seriousness about ensuring security and confidentiality within the Department.
Margaret Brennan sought to navigate this intense debate with finesse, often questioning the ethics and motives driving these policies.
She pondered if the very nature of these strategies reflected the administration’s approach to governance—a concern shared by many.
Political Discourse
This back-and-forth, while pointing fingers and raising questions, provided a window into the political dynamics surrounding U.S.
immigration policy. With both sides presenting their cases passionately, it was a classic example of the ongoing discourse within the nation, each defending their stance with fervor.
As we wrap up, it leaves us with much to ponder.
Is Noem's hard stance the answer, or does Brennan's appeal for a balanced approach carry more weight?
We'd love to hear what you think.
So, leave a comment below and share your perspective—how do you view Kristi Noem's policies and Brennan's critique?
Are we as a nation heading in the right direction? Let us know your thoughts!
"""

    
processed = process_transcript_with_chunks(transcript)
    

    
print(processed)
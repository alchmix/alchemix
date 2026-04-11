import time
import openai
import os
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, send_file
from threading import Thread
from gtts import gTTS
from tqdm import tqdm  
import threading
from dotenv import load_dotenv
# Initialize Flask app

load_dotenv()
# Upload Folder
app = Flask(__name__)
UPLOAD_FOLDER = "./tempstatic"
# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
FLICKR_API_KEY = os.getenv("FLICKR_API_KEY")
FLICKR_SECRET = os.getenv("FLICKR_SECRET")

# Initialize OpenAI Client (Optimize calls)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)



def generate_narration(script):
    # """ 
    # Cleans the script using GPT-4 and generates an audio narration with progress tracking. 
    # """

    # print("\n📝 Step 1: Cleaning Script using GPT-4...")
    # cleaned_script_prompt = f"""
    # I am about to create a narration of this script:
    
    # "{script}"
    
    # Clean it and return only the pure narration text that the audience will listen to.
    # """

    # # Track GPT-4 response time
    # start_time = time.time()
    # cleaned_script_response = openai_client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": cleaned_script_prompt}],
    #     temperature=0.3  # Faster, deterministic responses
    # )
    # cleaned_script = cleaned_script_response.choices[0].message.content.strip()
    # end_time = time.time()
    # print(f"✅ Script cleaned in {end_time - start_time:.2f} seconds.")

    # Generate unique filename for the narration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    narration_filename = f"narration_{timestamp}.mp3"
    narration_path = os.path.join(UPLOAD_FOLDER, narration_filename)

    # Step 2: Generating Audio Narration with Progress Tracking
    print("\n🎙️ Step 2: Generating Narration...")

    def save_audio():
        """ Function to generate and save the narration using gTTS. """
        tts = gTTS(script, lang="en", slow=False)
        tts.save(narration_path)

    # Use threading to allow progress tracking while generating narration
    thread = threading.Thread(target=save_audio)
    thread.start()

    # Display progress bar while narration is being generated
    with tqdm(total=100, desc="🔊 Saving Narration", unit="%", position=0, leave=True) as pbar:
        while thread.is_alive():
            pbar.update(10 if pbar.n < 90 else 100 - pbar.n)  # Gradually fill the bar
            time.sleep(0.5)  # Small delay for smooth progress
        pbar.update(100 - pbar.n)  # Ensure progress bar reaches 100%

    thread.join()  # Ensure thread completes

    print(f"\n✅ Narration successfully saved: {narration_path}")

    return narration_path, f"http://localhost:5000/tempstatic/{narration_filename}"

# script = """
# Today, we're diving into the hilariously turbulent world of Saturday Night Live, and guess what? It involves some of the biggest celebs that you might've totally forgotten once graced Studio 8H. But first, let's talk about a legendary meltdown that could have cost a career, but instead, became a footnote in the saga of one of comedy’s most iconic figures.

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


# print(generate_narration(script))
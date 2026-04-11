import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline, CLIPProcessor, CLIPModel

# Constants
def setup_nltk():
    nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))
    nltk.download('punkt', download_dir=os.path.join(os.getcwd(), "nltk_data"))
    nltk.download('averaged_perceptron_tagger', download_dir=os.path.join(os.getcwd(), "nltk_data"))
    nltk.download('averaged_perceptron_tagger_eng', download_dir=os.path.join(os.getcwd(), "nltk_data"))
    return True

if not setup_nltk():
    raise RuntimeError("Failed to initialize NLTK.")

def analyze_text_segments(narration_text):
    sentences = sent_tokenize(narration_text)
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    segments = []
    current_segment = []
    current_words = 0
    WORDS_PER_SEGMENT = 50

    for sentence in sentences:
        words = word_tokenize(sentence)
        current_words += len(words)
        current_segment.append(sentence)
        
        if current_words >= WORDS_PER_SEGMENT:
            segment_text = " ".join(current_segment)
            themes = classifier(segment_text, candidate_labels=["people", "location", "action", "emotion", "object"])
            tokens = nltk.pos_tag(word_tokenize(segment_text))
            key_terms = [word for word, pos in tokens if pos.startswith(('NN', 'NNP'))]
            estimated_duration = current_words * 0.4
            segments.append({
                'text': segment_text,
                'duration': estimated_duration,
                'themes': themes['labels'],
                'scores': themes['scores'],
                'key_terms': list(set(key_terms))
            })
            current_segment = []
            current_words = 0
    
    if current_segment:
        segment_text = " ".join(current_segment)
        themes = classifier(segment_text, candidate_labels=["people", "location", "action", "emotion", "object"])
        tokens = nltk.pos_tag(word_tokenize(segment_text))
        key_terms = [word for word, pos in tokens if pos.startswith(('NN', 'NNP'))]
        segments.append({
            'text': segment_text,
            'duration': current_words * 0.4,
            'themes': themes['labels'],
            'scores': themes['scores'],
            'key_terms': list(set(key_terms))
        })
    
    return segments

# narration_text = """Oh, get ready for this spicy scoop that’s steaming hot off the press! Picture this: New York City, the glistening glamor of Radio City Music Hall, and none other than the legendary Meryl Streep making waves in a way you might not expect. Now, when you think Meryl, you think class, exemplary talent, maybe a dash of regal poise, right? But, hold onto your hats because Meryl just served up a slice of shocker humor that's got everyone talking!

#         So here’s the scene—it’s February 14th, a night sparkling with stars, thanks to SNL’s 50th Homecoming Concert, and the cameras are rolling live. There, amidst the glitterati, we spot Meryl Streep cozying up next to her rumored beau and "Only Murders in the Building" co-star, Martin Short. Love is in the air, or so it seems, right? But wait, there’s a twist coming that’s as juicy as a ripe peach in summer!

#         Cut to Ana Gasteyer and Will Ferrell, I mean, c’mon, these two alone could stir up a laugh riot with just a glance! They’re back as our favorite dysfunctional middle school music teachers, Marty and Bobbi Mohan-Culp. Oh, how we've missed them! Will Ferrell, in his impeccable style, complains about seeing quite a few middle fingers from the crowd, calling it "unbelievably rude." And what do you know, the comic arrow turns to point straight at our dear Meryl.

#         Ana reels it in with a saucy, "We're talking about you, Meryl. We see you." And boom! Meryl, the queen herself, without missing a beat, flips the bird right back at them. Yes, you heard that right—Meryl Streep gave the finger! And it wasn’t just a coy little gesture. No, no, no. It was bold, it was brassy, and oh, did it get people howling with laughter, including Martin Short, whose reaction was just priceless!

#         Now, don’t go thinking Meryl’s just thrown decorum out the window. Oh no, this was all in good fun, a cheeky part of the gag lined up by the SNL alums, and it clearly stole the show. But isn't it just fabulous to see such an icon letting loose, causing a stir, and just generally embracing the mirth of the moment?

#         So there you have it, folks! Meryl Streep, flipping the script (and the bird) in a hilariously unexpected twist at an SNL celebration. Who says award-winning actresses can’t have a little naughty fun? Let’s just say, the audience at Radio City Music Hall sure got an eyeful and a playful reminder that Meryl is not only an on-screen legend but a bit of a live-wire jokester too! Meryl, you cheeky thing, you’ve done it again!"""
# results=analyze_text_segments(narration_text)
# print(results)

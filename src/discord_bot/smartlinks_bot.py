import discord
from openai import OpenAI
import os
from discord.ext import commands
import random
import time
import requests
import io
import pytesseract
from PIL import Image
from pathlib import Path
import aiohttp
import PyPDF2
import functools
import typing
import asyncio
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import textwrap
from pytube import YouTube
import moviepy.editor as mp
import os
import stat
import tempfile
from pydub import AudioSegment
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from youtube_transcript_api import YouTubeTranscriptApi
from whisper_timestamped import transcribe
import whisper
import re
from bs4 import BeautifulSoup
import json

import os
from dotenv import load_dotenv

# Load your Discord bot token and OpenAI API key from environment variables
load_dotenv()  # take environment variables from .env.
DISCORD_TOKEN = os.getenv('SMART_LINK_DISCORD_TOKEN')
# DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

global VERSION
VERSION = "0.2.1-hn"

client = OpenAI(api_key=OPENAI_API_KEY)

# Define bot command prefix
COMMAND_PREFIX = '!'

# Initialize Discord bot
intents = discord.Intents.default()
intents.message_content = True
intents.typing = True  # Enable if necessary
intents.presences = True  # Enable if necessary
bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)
dsclient = discord.Client(intents=intents)


def chatgpt(prompt="", model="gpt-3.5-turbo"):
    """Call chatgpt with a prompt and get a message returned."""
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return chat_completion.choices[0].message.content

def chatgpt_v2(system_prompt="", user_content="", model="gpt-3.5-turbo"):
    """Call chatgpt with a prompt and get a message returned."""
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    )
    return chat_completion.choices[0].message.content

@bot.event
async def on_ready():
    msg = f'{bot.user.name} has connected to Discord! Version: {VERSION} | Type `!info` for a list of commands.'

@bot.command(name="i", aliases=["info"])
async def info(ctx):
    ex_user_prompt = "`<extracted_text>{text}</extracted_text>`"
    extra = """
    Description:
        Share screenshots of social media posts, academic papers, podcasts, and YouTube videos.
        Get short summaries/analysis to figure out if you should invest more time into the content.

        `!a` is the Dicord bot command (short for analyze) to download, transcribe and analyze the text content.
        The default system prompt is an opinionated analysis prompt customized for each media type.
        You can override the default system prompt with your own prompt by adding it as the last argument.
        
        The following string will be prepended to the override_system_prompt:
        `You will be provided with a chunk of text (delimited with XML tags) from` `INPUT YOUR CUSTOM PROMPT HERE`

        The parsed text of the media inserts into the first user_prompt like so:
        {ex_user_prompt}.
    """
    help_msg = textwrap.dedent(
        f"""
        Discord Bot Version: {VERSION}
        Save time and share ideas more effectively by pre-processing hypermedia content with a short command.

        Usage: 
        `!a [media_type] [media_url] [override system prompt]`

        Available Media Types:
        `['img', 'mp3', 'yt (YouTube)', 'pdf', 'hn (HackerNews Comments)']`
        Planned Media Types
        `['web', 'rd (Reddit Posts/Comments)']`
        
        Options:
        `!i`,  Show this info message.

        Examples:
        `!a pdf www.pdfs.com/filename.pdf`
        `!a yt www.youtube.com/123video`
        `!a mp3 www.mp3host.com/filename.mp3`
        `!a img` (with attached image)
        `!a hn "Why are we templating YAML? (2019) - https://news.ycombinator.com/item?id=39101828"`

        Override System Prompt Examples:
        `!a pdf www.linktopdf.com/file.pdf a pdf containing an academic paper. Rewrite the content in the style of a pirate.`
        `!a yt www.youtube.com/123video a video transcript. Rewrite the content in the style of a pirate.`
        `!a img ANYWORDFORTHEURL social media posts. Rewrite the content in the style of a pirate.`
        """
    )
    await ctx.send(help_msg)
    

@bot.command(name="a", aliases=["general_analyze"])
async def general_analyze(ctx, media_type=None, input_url=None, *, override_system_prompt=None):
    valid_media_types = ["web", "txt", "yt", "img", "imgl", "pdf", "mp3", "rd", "hn"]

    ex_args = "`messages=[{'role': 'user', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]`"
    ex_user_prompt = "`<extracted_text>{text}</extracted_text>`"
    error_message = textwrap.dedent(
        f"""
        Discord Bot Version: {VERSION}
        Command Structure:
        `!a $MEDIA_TYPE $URL_TO_MEDIA $OPTIONAL_SYSTEM_PROMPT_OVERRIDE`
        
        {valid_media_types=}

        The parsed text of the media inserts into the first user_prompt like so:
        {ex_user_prompt}.
        
        Use the OPTIONAL_SYSTEM_PROMPT_OVERRIDE to set your own system prompt analysis instructions. 
        By default, the following string will be prepended to the `$OPTIONAL_SYSTEM_PROMPT_OVERRIDE`:
        `You will be provided with a chunk of text (delimited with XML tags) from` 
        
        Example messages argument:
        {ex_args}
        
        Example Commands:
        `!a img` (with attached image)

        Example Override System Prompt Commands:
        `!a pdf www.linktopdf.com/file.pdf a pdf containing an academic paper. Rewrite the content in the style of a pirate.`

        `!a yt www.youtube.com/123video a video transcript. Rewrite the content in the style of a pirate.`

        `!a img ANYWORDFORTHEURL social media posts. Rewrite the content in the style of a pirate.`
        """)
    
    supported_types_error_msg = "\n\nSupported types:\n`web`, `txt`, `yt`, `img`, `imglink`, `pdf`, `mp3`, `rd`, `hn`\n"
    
    # Check if the media_type is provided
    if not media_type:
        await ctx.send(f"Missing media type.{supported_types_error_msg}{error_message}")
        return
    # Check if the media_type is valid
    if media_type not in valid_media_types:
        await ctx.send(f"Invalid media type.{supported_types_error_msg}{error_message}")
        return
    # Check if the input_url is provided
    if not input_url and media_type not in ["txt", "img"]:
        await ctx.send(f"Missing input url.\n\nPlease provide a URL to the content.{error_message}")
        return
    
    # Check for supported media types and extract content
    # each one will have its own media_to_text function
    media_type = media_type.lower()
    if media_type == "web":
        # Handle raw website content
        pass
    elif media_type == "txt": # MIGHT NEED TO BE ITS OWN COMMAND
        # Handle text content
        pass
    elif media_type == "yt":

        # try to extract the automatic transcript from the video
        subtitle_text, video_length = await extract_youtube_subtitles_v2(input_url)

        if subtitle_text:
            await ctx.send(f"This video's runtime is {video_length}. Analysis in progress...\n")
            try:
                for i in range(0, len(subtitle_text), 40000):
                    response = await _generate_yt_summary_v2(subtitle_text, i)
                    msg = response.choices[0].message.content
                    for k in range(0, len(msg), 1998):
                        await ctx.send(msg[k:k+1998])
            except Exception as e:
                print(f"Error: {e}")
                await ctx.send("I am sorry, but I cannot respond to that right now.")

        # if no transcript, download the video and extract the audio
        elif not subtitle_text:
            await ctx.send(f"No automatic transcript exists. Download and transcription in progress...\n")
            video_path, title = await _download_youtube_video_v2(input_url)
            subtitle_text, video_length = await _generate_transcript_v2(video_path)
            await ctx.send(f"This video's runtime is {video_length}. Analysis in progress...\n")
            try:
                for i in range(0, len(subtitle_text), 40000):
                    response = await _generate_yt_summary_v2(subtitle_text, i)
                    msg = response.choices[0].message.content
                    for k in range(0, len(msg), 1998):
                        await ctx.send(msg[k:k+1998])
            except Exception as e:
                print(f"Error: {e}")
                await ctx.send("I am sorry, but I cannot respond to that right now.")

    elif media_type == "img":
        # Handle image content
        # Check if an image was sent
        if ctx.message.attachments:
            attachment = ctx.message.attachments[0]  # Get the first attachment
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif']):
                image_bytes = await attachment.read()  # Read the image data
                image_file = io.BytesIO(image_bytes)

                # Use PIL to open the image
                image = Image.open(image_file)

                text = pytesseract.image_to_string(image)
                print(f"{len(text)=}")

                base_system_prompt_prefix = "You will be provided with a chunk of text (delimited with XML tags) from"
                system_prompt = "OCR of screenshots containing partial social media forum posts that may contain multiple writers. Identify the person or people (if its a multi user forum post screenshot ocr) who posted the content, summarize the main concepts, arguments, and critical pieces of information of the content. If there are step by step instructions, recommended actions, or other resources like books or links in the post include them in your output. If there are enumerated lists, include something about each item in the list in your output. Always use common, straight forward words in your output following Amazon's writing principles. Finally, write a single short sentence summarizing why someone should invest more time into reading more about the content at the very end by itself on a new line using the principles of the book Sell or Be Sold. Please keep the whole thing under 1500 characters and minimal new line characters. Use informationally dense words; write like Ernest Hemingway trained as a technical writer so exclude any filler, introductionary-esque pieces of any output."
                user_prompt = f"<extracted_text>{text}</extracted_text>"

                if override_system_prompt:
                    system_prompt = base_system_prompt_prefix + " " + override_system_prompt
                else:
                    # fullprompt = f"Based on the input text, Summarize the key points succinctly within 150 words, and speculate on the potential outcomes or next steps discussed in the text within another 150 words. Formulate 3 short questions/problems the text helps answer/solve and a short no fluff, specific reason why someone to invest more time into it, like what problem will it help solve?\nINPUT_TET: {text}"
                    system_prompt = base_system_prompt_prefix + " " + system_prompt
                    user_prompt = f"<extracted_text>{text}</extracted_text>"
                
                print(f"{system_prompt=}")
                print(f"{user_prompt=}")
                try:
                    response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                            )
                    print(response.choices[0].message.content)
                    
                    responsetext = response.choices[0].message.content
                    for i in range(0, len(responsetext), 1999):
                        await ctx.send(responsetext[i:i+1999]) 
                except Exception as e:
                    print(f"Error: {e}")
                    await ctx.send("I am sorry, but I cannot respond to that right now.") 
            else:
                await ctx.send('Please upload an image.')
        else:
            await ctx.send('Please attach an image.')
    elif media_type == "imglink":
        # Handle image link content
        pass
    elif media_type == "pdf":
        # Handle PDF content
        try:
            # Use aiohttp to fetch the PDF content
            async with aiohttp.ClientSession() as session:
                async with session.get(input_url) as response:
                    if response.status == 200:
                        pdf_bytes = await response.read()
                        pdf_file = io.BytesIO(pdf_bytes)
                        
                        # Use PyPDF2 to read the PDF file
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        text = ""
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            text += page.extract_text()

                        await ctx.send(f"This paper, {input_url} has {len(pdf_reader.pages)} pages and {len(text)} total characters.")
                        
                        abstract_index = min(text.lower().find('abstract'), text.lower().find('intro'))
                        conclusion_index = max(text.lower().find('conclusion\n\n'), text.lower().find('concluding remarks\n\n'))

                        if abstract_index == -1:
                            abstract_index = 0
                        if conclusion_index == -1:
                            conclusion_index = text.lower().find('conclud')
                            if conclusion_index == -1:
                                conclusion_index = text.lower().find('discussion')

                        
                        base_system_prompt_prefix = "You will be provided with a chunk of text (delimited with XML tags) from"
                        system_prompt = "PDF files of a book or an academic paper. Summarize the main concepts, arguments, and critical pieces of information of the content, and speculate on the potential applications or further areas of study. Formulate 3 short questions/problems the paper helps answer/solve. Always use common, straight forward words in your output following Amazon's writing principles. Finally, write a single short sentence summarizing why someone should invest more time into reading more about the content at the very end by itself on a new line using the principles of the book Sell or Be Sold. Please keep the whole thing under 1500 characters and minimal new line characters. Use informationally dense words; write like Ernest Hemingway trained as a technical writer so exclude any filler, introductionary-esque pieces of any output."
                        user_prompt = f"<extracted_text>{text}</extracted_text>"

                        if override_system_prompt:
                            system_prompt = base_system_prompt_prefix + " " + override_system_prompt
                        else:
                            system_prompt = base_system_prompt_prefix + " " + system_prompt
                            user_prompt = f"<extracted_text>{text[abstract_index:abstract_index+1250]} {text[conclusion_index:conclusion_index+1250]}</extracted_text>"
                        
                        print(f"{system_prompt=}")
                        print(f"{user_prompt=}")

                        chatbotmessages = [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ]
                        response = client.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=chatbotmessages
                                )
                        
                        # Send the API response back to the Discord channel
                        await ctx.send(response.choices[0].message.content[:1999])
                    else:
                        await ctx.send("Failed to download the PDF. Please check the URL.")
        except Exception as e:
            await ctx.send(f"An error occurred: {e}")
    elif media_type == "mp3":
        # handle MP3 content
        try:
            response = requests.get(input_url)
            if response.status_code == 200:                
                # Save the MP3 file temporarily, you might want to handle the file path more carefully in production
                mp3_bytes = response.content
                file_path = 'podcast.mp3'
                with open(file_path, 'wb') as mp3_file:
                    mp3_file.write(mp3_bytes)
                
                # Assuming you have Whisper installed and imported
                # Use Whisper to transcribe the MP3 file
                await ctx.send(f'Transcribing audio of {input_url}')
                transcript = await _generate_transcript()

                await ctx.send(f"This mp3, {input_url} has {len(transcript)} total characters.")

                # Send the transcript back to the Discord channel
                # Discord messages have a character limit, so you may need to split long transcripts
                for i in range(0, len(transcript), 45000):
                    response = await _generate_mp3_summary(transcript, i, override_system_prompt)
                    await ctx.send(response.choices[0].message.content[:1999])

            else:
                await ctx.send("Failed to download the MP3. Please check the URL.")
        except Exception as e:
            await ctx.send(f"An error occurred: {e}")
    elif media_type == "rd":
        # handle Reddit content
        pass
    elif media_type == "hn":

        hn_url = re.search(r'https://news\.ycombinator\.com/item\?id=\d+', input_url)
        hn_url = hn_url.group(0)

        # download and return hackernews curl html string
        response = requests.get(hn_url)
        html_content = response.text

        # send processing message with length of comments to process.
        await ctx.send(f"Processing {len(html_content)} characters of HTML content.")

        hn_comments_json_string = await _get_hn_comments(html_content)

        final_response = await _process_hn_curl_html_string(hn_comments_json_string)

        for i in range(0, len(final_response), 1999):
            await ctx.send(final_response[i:i+1999])

    else:
        # Handle unsupported media type
        pass


def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper

@to_thread
def _get_hn_comments(html_string):
    print("starting bs4")
    soup = BeautifulSoup(html_string, 'html.parser')
    comments = []

    comment_elements = soup.find_all(class_='comtr')
    for comment_element in comment_elements:
        username_element = comment_element.find(class_='hnuser')
        comment_text_element = comment_element.find(class_='commtext')

        # Check if both username and comment text elements are found
        if username_element and comment_text_element:
            username = username_element.text
            comment_text = comment_text_element.text

            comment_data = {
                'username': username,
                'comment': comment_text
            }
            comments.append(comment_data)

    document_string = json.dumps(comments, indent=2)

    print(f"{len(document_string)=}")
    return document_string

@to_thread
def _process_hn_curl_html_string(html_comments_string):
    document_string = html_comments_string
    chunk_size = 40000
    chunk_responses = []

    if len(document_string) > chunk_size:

        for i in range(0, len(document_string), chunk_size):
            chunk_response = chatgpt_v2(
                system_prompt="You will receive a list of json comments from a HackerNews thread. Extract and explain the discussed problems, philosophical ideas and concepts, arguments, and perspectives the users have. ALWAYS INCLUDE any relevant links to outside resources recommended by users in a list. Use informationally dense, but clear technical writing style without filler or transitional words and phrases.",
                user_content=document_string[i: i+chunk_size],
                model="gpt-3.5-turbo-16k"
            )
            chunk_responses.append(chunk_response)
            print(f"processed {i+chunk_size} of {len(document_string)} characters")
            time.sleep(1.6)

        final_response = chatgpt_v2(
            system_prompt="You will receive a list of summarized HackerNews discussion threads in terms of concepts, arguments, and perspectives users have collectively. Concatenate/merge the outputs to be within 1500 characters and include the list of external links recommended by users.",
            user_content=str(chunk_responses),
            model="gpt-3.5-turbo-16k"
        )
        print(chunk_responses)
        print()
        print(final_response)
    else:
        final_response = chatgpt_v2(
            system_prompt="You will receive a list of json comments from a HackerNews thread. Extract and explain the discussed problems, philosophical ideas and concepts, arguments, and perspectives the users have. Include any relevant links to outside resources recommended by users in a list. Use informationally dense, but clear technical writing style without filler or transitional words and phrases.",
            user_content=document_string,
            model="gpt-3.5-turbo-16k"
        )
        print(f"processed {chunk_size} of {len(document_string)} characters")
        print(final_response)

    return final_response

@to_thread
def _get_list_of_chunked_yt_file_paths(input_url):
    try:
        yt = YouTube(input_url)
        video_stream = yt.streams.filter(only_audio=True, abr="128kbps").first()

        if video_stream:
                    # Create a temporary directory for storing the video and audio files
            mp3_dir = Path.cwd() / "temp"
            mp3_dir.mkdir(exist_ok=True)

                    # Generate a unique temporary file name for the video
            mp3_file = "tempfile.mp3"
                    
                    # Download the video to the temporary file
            video_stream.download(output_path=mp3_dir, filename=mp3_file)

            mp3_fullpath = mp3_dir / mp3_file
            print(f"{mp3_fullpath=}")

            audio_segment = AudioSegment.from_file(mp3_fullpath)
            audio_segment = audio_segment.set_frame_rate(44100).set_channels(2)
            audio_segment.export(mp3_fullpath, format="mp3")

            mp3 = AudioSegment.from_mp3(mp3_fullpath)

            twenty_min = 20 * 60 * 1000

            filenum = 0
            chunked_mp3_paths = []
            for i in range(0, len(mp3), twenty_min):
                fp = mp3_dir / f"tempfile{filenum}.mp3"
                mp3[i:i+twenty_min].export(fp, format="mp3")
                filenum += 1
                chunked_mp3_paths.append(fp)
            return chunked_mp3_paths, len(mp3)
        else:
            print("No audio stream found in the video.")
            return None, None
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None, None
    

@to_thread
def _generate_yt_summary(transcript, i, override_system_prompt=None):
    base_system_prompt_prefix = "You will be provided with a chunk of text (delimited with XML tags) from"
    system_prompt = "a YouTube video subtitles or transcript of either a monologue or a podcast. Summarize the main concepts, arguments, and critical pieces of information from the video. Formulate 3 short questions/problems the video content helps answer/solve. Always use common, straightforward words in your output following Amazon's writing principles. Finally, write a single short sentence summarizing why someone should invest more time in watching the video. Please keep the whole thing under 1500 characters and minimal new line characters. Use informationally dense words; write like Ernest Hemingway trained as a technical writer so exclude any filler, introductory-esque pieces of any output. Always remember to end the output with the final summary sentence following the previous instructions."
    user_prompt = f"<extracted_text>{transcript[i:i+45000]}</extracted_text>"

    if override_system_prompt:
        system_prompt = base_system_prompt_prefix + " " + override_system_prompt
    else:
        # fullprompt = f"Based on the input text, Summarize the key points succinctly within 150 words, and speculate on the potential outcomes or next steps discussed in the text within another 150 words. Formulate 3 short questions/problems the text helps answer/solve and a short no fluff, specific reason why someone to invest more time into it, like what problem will it help solve?\nINPUT_TET: {text}"
        system_prompt = base_system_prompt_prefix + " " + system_prompt
        user_prompt = f"<extracted_text>{transcript}</extracted_text>"
    
    print(f"{system_prompt=}")
    print(f"{user_prompt=}")
    response = client.chat.completions.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}]
                )
    
    return response


@to_thread
def _generate_yt_summary_v2(transcript, i):
    # output_structure = textwrap.dedent("""
    #     Topic Name (start_time - end_time)                                       
    #     Topic Name (start_time - end_time)                                     
    #     Topic Name (start_time - end_time)

    #     Short Summary:
    #     """)
    # system_prompt = f"Semantically chunk the following transcript wrapped in XML tags into topic segments. For each topic, please include the timestamp where it began in the transcript. At the end of the topic segmentation, include a short summary. Please keep the entire analysis under 1500 characters. Always include the timestamps of the topics."

    output_structure = textwrap.dedent("""
        Topic Name (start_time - end_time)                                       
        Topic Name (start_time - end_time)                                     
        Topic Name (start_time - end_time)

        Short Summary:
        """)
    system_prompt = f"Semantically segment the following transcript wrapped in XML tags into high level topics with a 1 sentence extractive summary that include critical information, any arguments or claims they are making, supporting resources like links to books or other videos, or any calls to action except for liking/subscribing to the video. Use informationally dense words and do not include any fluffy filler words. Write like profesionnal technical writer. The ultimately purpose is to figure out if the video is worth watching. For each topic, please include the timestamp where it began in the transcript. Do NOT output more than 1500 characters, the user will not be able to see it. Do NOT use bullet points. Use this formatting '[starttime] TOPICNAMEHERE' style for each topic. Make sure the timestamps are accurate."
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"<video_transcript>{transcript[i:i+40000]}</video_transcript>"}]
    response = client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=messages
            )
    print(messages)
    print(response)
    return response

# @to_thread
# def _generate_yt_summary_v2(transcript, i):
#     output_structure = textwrap.dedent("""
#         Topic Name: Topic Title (start_time - end_time)
#           Sub Topic Name (start_time - end_time)
#           Sub Topic Name (start_time - end_time)
                                       
#         Topic Name: Topic Title (start_time - end_time)
#           Sub Topic Name (start_time - end_time)
#           Sub Topic Name (start_time - end_time)
                                       
#         Topic Name: Topic Title (start_time - end_time)
#           Sub Topic Name (start_time - end_time)
#           Sub Topic Name (start_time - end_time)

#         Entire Short Summary:
#         """)
#     system_prompt = f"Semantically chunk the following full or partial transcript JSON into segments by topic following this desired output structure, '''{output_structure}'''. Replace 'Sub Topic Name' with the detected name of each subtopic. Replace 'Entire Short Summary:' with just the 3 sentence summary. Replace 'Topic Name:' with just the detected topic name. Please keep the entire analysis under 1500 characters. Always include the timestamps of the topics."
#     response = client.chat.completions.create(
#                 model="gpt-3.5-turbo-16k",
#                 messages=[{"role": "system", "content": system_prompt},
#                         {"role": "user", "content": transcript[i:i+45000]}]
#             )
#     return response

@to_thread
def _generate_mp3_summary(transcript, i, override_system_prompt=None):
    base_system_prompt_prefix = "You will be provided with a chunk of text (delimited with XML tags) from"
    system_prompt = "a transcript of an mp3 podcast. Identify the person or people (if its a multi partipant podcast) who posted the content, Summarize the main concepts, arguments, and critical pieces of information of the content, and speculate on the potential applications or further areas of study. Formulate 3 short questions/problems the paper helps answer/solve. Always use common, straight forward words in your output following Amazon's writing principles. Finally, write a single short sentence summarizing why someone should invest more time into reading more about the content at the very end by itself on a new line using the principles of the book Sell or Be Sold. Please keep the whole thing under 1500 characters and minimal new line characters. Use informationally dense words; write like Ernest Hemingway trained as a technical writer so exclude any filler, introductionary-esque pieces of any output."
    # system_prompt = "Based on the input text, Summarize the key points, actionable takeaways succinctly within 150 words, and speculate on the potential outcomes or next steps discussed in the text within another 150 words. Formulate 3 short questions/problems the text helps answer/solve and a short no fluff, specific reason why someone to invest more time into it, like what problem will it help solve?"
    user_prompt = f"<extracted_text>{transcript[i:i+45000]}</extracted_text>"

    if override_system_prompt:
        system_prompt = base_system_prompt_prefix + " " + override_system_prompt
    else:
        # fullprompt = f"Based on the input text, Summarize the key points succinctly within 150 words, and speculate on the potential outcomes or next steps discussed in the text within another 150 words. Formulate 3 short questions/problems the text helps answer/solve and a short no fluff, specific reason why someone to invest more time into it, like what problem will it help solve?\nINPUT_TET: {text}"
        system_prompt = base_system_prompt_prefix + " " + system_prompt
        user_prompt = f"<extracted_text>{transcript}</extracted_text>"
    
    print(f"{system_prompt=}")
    print(f"{user_prompt=}")
    response = client.chat.completions.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}]
                )
    
    return response

@to_thread
def _consolidate_yt_summaries(summaries_list):
    system_prompt = "You will be provided with the output of summaries generated from partial segments of the audio of a youtube containing a podcast with multiple people or a monologue (delimited with XML tags). The original prompts that made each 'segment_i_summary' came from the following original prompt. <original_summary_prompt>Identify the person or people (if its a multi partipant podcast) who posted the content, Summarize the main concepts, arguments, and critical pieces of information of the content, and speculate on the potential applications or further areas of study. Formulate 3 short questions/problems the paper helps answer/solve. Always use common, straight forward words in your output following Amazon's writing principles. Finally, write a single short sentence summarizing why someone should invest more time into reading more about the content at the very end by itself on a new line using the principles of the book Sell or Be Sold. Please keep the whole thing under 1500 characters and minimal new line characters. Use informationally dense words; write like Ernest Hemingway trained as a technical writer so exclude any filler, introductionary-esque pieces of any output.</original_summary_prompt> You are to combine each of the summaries to consolidate all of the unique information into a singular output, while keeping the same final output structure from the original prompt."
    user_prompt = ""
    for i, summ in enumerate(summaries_list):
        user_prompt += f"<segment_{i+1}_summary>{summ}</segment_{i+1}_summary>"
    
    print(f"{system_prompt=}")
    print(f"{user_prompt=}")
    response = client.chat.completions.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}]
                )
    
    return response

@to_thread
def _generate_transcript(filename="podcast.mp3"):
    with open(filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file, 
                    response_format="text"
                )
        
    return transcript

@to_thread
def _generate_transcript_v2(filename):
    def decimal_to_time(decimal_time):
        """Converts decimal time to time format."""
        total_seconds = int(decimal_time)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours == 0:
            time_str = f"{minutes:02}:{seconds:02}"
        else:
            time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        return time_str

    model = whisper.load_model("base")
    result = transcribe(model, str(filename))

    formatted_transcript = []
    for segment in result['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        formatted_transcript.append({
            'text': text,
            'start': start_time,
            'duration': round(end_time - start_time, 2)
        })

    runtime = decimal_to_time(formatted_transcript[-1]["start"])
    short_transcript = "\n".join([f"{decimal_to_time(x['start'])}: '{x['text']}'" for x in formatted_transcript])

    return short_transcript, runtime

@to_thread
def _download_youtube_video_v2(video_url):
    yt = YouTube(video_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    mp3_dir = Path.cwd() / "temp"
    mp3_dir.mkdir(exist_ok=True)
    mp3_full_path =mp3_dir / 'downloaded_audio.mp3'
    audio_stream.download(filename=mp3_full_path)
    return mp3_full_path, yt.title


@to_thread
def extract_youtube_subtitles_v2(video_url):

    def decimal_to_time(decimal_time):
        """Converts decimal time to time format."""
        total_seconds = int(decimal_time)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours == 0:
            time_str = f"{minutes:02}:{seconds:02}"
        else:
            time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        return time_str

    try:
        match = re.search(r"(?<=v=)[^&#]+", video_url)
        video_id = match.group() if match else None
        print(f"{video_id=}")

        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
        short_transcript = [f"{decimal_to_time(x['start'])}: '{x['text']}'" for x in transcript]
        short_transcript = "\n".join(short_transcript)

        video_length = decimal_to_time(transcript[-1]["start"])

        print(f"{short_transcript=}")

        return short_transcript, video_length

    except Exception as e:
        print(f"Error extracting subtitles: {e}")
        return None, None

@to_thread
def extract_youtube_subtitles(video_url):
    try:
        yt = YouTube(video_url)
        captions = yt.captions.all()
        
        if captions:
            subtitle = captions[0]
            subtitle_text = subtitle.generate_srt_captions()
            return subtitle_text
        else:
            return None
    except Exception as e:
        print(f"Error extracting subtitles: {e}")
        return None

@to_thread
def download_and_transcribe_youtube_video(video_url):
    try:
        yt = YouTube(video_url)
        video_stream = yt.streams.filter(only_audio=True).first()

        if video_stream:
            # Create a temporary directory for storing the video and audio files
            temp_dir = Path.cwd() / "temp"
            temp_dir.mkdir(exist_ok=True)

            
            # Generate a unique temporary file name for the video
            video_filename = "tempfile.mp4"
            video_path = temp_dir / video_filename
            
            # Download the video to the temporary file
            video_stream.download(output_path=video_path, filename=video_filename)
            
            audio_filename = "tempfile.mp3"
            audio_path = video_path.parent / audio_filename

            video_clip = mp.VideoFileClip(str(video_path))
            audio_clip = video_clip.audio
            
            # Write the audio to the temporary file
            audio_clip.write_audiofile(audio_path)
            return audio_path
        else:
            print("No audio stream found in the video.")
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None


if __name__ == '__main__':
    bot.run(DISCORD_TOKEN)
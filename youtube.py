import streamlit as st
from pytube import YouTube
import os
import requests
from time import sleep
import shutil
import librosa
import openai
import soundfile as sf
import youtube_dl
from youtube_dl.utils import DownloadError
import yt_dlp as youtube_dl
import pickle
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
from dotenv import load_dotenv
import base64
from langchain.memory import ConversationSummaryMemory
from langchain.chains import (
     LLMChain, ConversationalRetrievalChain
)
from langchain_groq import ChatGroq
from st_clickable_images import clickable_images
import pandas as pd

llm = ChatGroq(temperature=0, groq_api_key="groq_api_key", model_name="mixtral-8x7b-32768")

OPENAI_API_KEY = "openai_api_key"
openai.api_key = OPENAI_API_KEY

upload_endpoint = "https://api.assemblyai.com/v2/upload"
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

headers = {
    "authorization": "assemblyai_api_key",
    "content-type": "application/json"
}

def save_audio(url):
    yt = YouTube(url)
    try:
        video = yt.streams.filter(only_audio=True).first()
        out_file = video.download()
    except:
        return None, None, None
    base, ext = os.path.splitext(out_file)
    file_name = base + '.mp3'
    os.rename(out_file, file_name)
    print(yt.title + " has been successfully downloaded.")
    print(file_name)
    return yt.title, file_name, yt.thumbnail_url

def upload_to_AssemblyAI(save_location):
    CHUNK_SIZE = 5242880
    print(save_location)

    def read_file(filename):
        with open(filename, 'rb') as _file:
            while True:
                print("chunk uploaded")
                data = _file.read(CHUNK_SIZE)
                if not data:
                    break
                yield data

    upload_response = requests.post(
        upload_endpoint,
        headers=headers, data=read_file(save_location)
    )
    print(upload_response.json())

    if "error" in upload_response.json():
        return None, upload_response.json()["error"]

    audio_url = upload_response.json()['upload_url']
    print('Uploaded to', audio_url)

    return audio_url, None

def start_analysis(audio_url):
    print(audio_url)

    ## Start transcription job of audio file
    data = {
        'audio_url': audio_url,
        'iab_categories': True,
        'content_safety': True,
        "summarization": True,
        "summary_model": "informative",
        "summary_type": "bullets"
    }

    transcript_response = requests.post(transcript_endpoint, json=data, headers=headers)
    print(transcript_response.json())

    if 'error' in transcript_response.json():
        return None, transcript_response.json()['error']

    transcript_id = transcript_response.json()['id']
    polling_endpoint = transcript_endpoint + "/" + transcript_id

    print("Transcribing at", polling_endpoint)
    return polling_endpoint, None

def get_analysis_results(polling_endpoint):
    status = 'submitted'

    while True:
        print(status)
        polling_response = requests.get(polling_endpoint, headers=headers)
        status = polling_response.json()['status']

        if status == 'submitted' or status == 'processing' or status == 'queued':
            print('not ready yet')
            sleep(10)

        elif status == 'completed':
            print('creating transcript')
            return polling_response

        else:
            print('error')
            return False

def find_audio_files(path, extension=".mp3"):
    """Recursively find all files with extension in path."""
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(extension):
                audio_files.append(os.path.join(root, f))

    return audio_files

def youtube_to_mp3(youtube_url: str, output_dir: str):
    """Download the audio from a youtube video, save it to output_dir as an .mp3 file.

    Returns the filename of the saved video.
    """

    ffmpeg_path = "C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe"  # Specify the path to FFmpeg

    # Config
    ydl_config = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "verbose": True,
        "ffmpeg_location": ffmpeg_path  # Add this line
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])
    except DownloadError:
        # weird bug where youtube-dl fails on the first download, but then works on second try... hacky ugly way around it.
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])

    audio_filename = find_audio_files(output_dir)[0]
    return audio_filename

def chunk_audio(filename, segment_length: int, output_dir):
    """Segment length is in seconds"""

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Load audio file
    audio, sr = librosa.load(filename, sr=44100)

    # Calculate duration in seconds
    duration = librosa.get_duration(y=audio, sr=sr)

    # Calculate number of segments
    num_segments = int(duration / segment_length) + 1

    # Iterate through segments and save them
    for i in range(num_segments):
        start = i * segment_length * sr
        end = (i + 1) * segment_length * sr
        segment = audio[start:end]
        sf.write(os.path.join(output_dir, f"segment_{i}.mp3"), segment, sr)

    chunked_audio_files = find_audio_files(output_dir)
    return sorted(chunked_audio_files)

def transcribe_audio(audio_files: list, output_file=None, model="whisper-1"):
    transcripts = []
    for audio_file in audio_files:
        audio = open(audio_file, "rb")
        response = openai.Audio.transcribe(model, audio)
        transcripts.append(response["text"])

    if output_file is not None:
# Save all transcripts to a .txt file
        with open(output_file, "w") as file:
            for transcript in transcripts:
                file.write(transcript + "\n")

    return transcripts

def get_transcript(youtube_url: str, outputs_dir: str):
    raw_audio_dir = f"{outputs_dir}/raw_audio/"
    chunks_dir = f"{outputs_dir}/chunks"
    transcripts_file = f"{outputs_dir}/transcripts.txt"
    summary_file = f"{outputs_dir}/summary.txt"
    # Download audio from youtube
    audio_filename = youtube_to_mp3(youtube_url, outputs_dir)
    segment_length = 10 * 60
    
    audio_filename =  youtube_to_mp3(youtube_url,raw_audio_dir)
    chunked_audio_files = chunk_audio(
        audio_filename, segment_length=segment_length, output_dir=chunks_dir
    )
    transcriptions = transcribe_audio(chunked_audio_files, transcripts_file)
    return transcriptions

def create_vector_store(youtube_link):
    st.write("Downloading the YouTube video...")
    transcripts = get_transcript(youtube_link, "outputs")
    doc = Document(page_content="\n".join(transcripts), metadata={"source": youtube_link})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    chunks = text_splitter.split_documents([doc])
    print(chunks)
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    db = Chroma.from_documents(chunks, embeddings, persist_directory=f"./new_db_{youtube_link.replace('https://www.youtube.com/watch?v=', '')}")
    db.persist()
    
def clear_folders():
    audio_dir = "audio"
    output_dir = "outputs"
    if os.path.exists(audio_dir):
        shutil.rmtree(audio_dir)
    os.makedirs(audio_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    if "vector_store_created" in st.session_state:
        del st.session_state.vector_store_created
        st.experimental_rerun()


def main():
    pages = {
        "Video Summary": video_summary,
        "Video Q&A": video_qa,
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selection]()

def video_summary():
    st.header("📄Video Summary and Analysis")
    youtube_link = st.text_input("Paste your YouTube link here")
    if youtube_link:
        try:
            video_title, save_location, video_thumbnail = save_audio(youtube_link)
            if video_title:
                st.header(video_title)
                st.audio(save_location)

                # upload mp3 file to AssemblyAI
                audio_url, error = upload_to_AssemblyAI(save_location)
                
                if error:
                    st.write(error)
                else:
                    # start analysis of the file
                    polling_endpoint, error = start_analysis(audio_url)

                    if error:
                        st.write(error)
                    else:
                        # receive the results
                        results = get_analysis_results(polling_endpoint)

                        summary = results.json()['summary']
                        topics = results.json()['iab_categories_result']['summary']
                        sensitive_topics = results.json()['content_safety_labels']['summary']

                        st.header("Summary of this video")
                        st.write(summary)

                        st.header("Sensitive content")
                        if sensitive_topics != {}:
                            st.subheader('🚨 Mention of the following sensitive topics detected.')
                            moderation_df = pd.DataFrame(sensitive_topics.items())
                            moderation_df.columns = ['topic','confidence']
                            st.dataframe(moderation_df, use_container_width=True)

                        else:
                            st.subheader('✅ All clear! No sensitive content detected.')

                        st.header("Topics discussed")
                        topics_df = pd.DataFrame(topics.items())
                        topics_df.columns = ['topic','confidence']
                        topics_df["topic"] = topics_df["topic"].str.split(">")
                        expanded_topics = topics_df.topic.apply(pd.Series).add_prefix('topic_level_')
                        topics_df = topics_df.join(expanded_topics).drop('topic', axis=1).sort_values(['confidence'], ascending=False).fillna('')

                        st.dataframe(topics_df)

        except Exception as e:
            st.error(f"An error occurred: {e}")

def video_qa():
    st.header("📄Chat with content from a YouTube video🤗")
    # Upload a YouTube link
    clear_button = st.button("Clear Folders and Reset")
    if clear_button:
        clear_folders()
    youtube_link = st.text_input("Paste your YouTube link here")
    if youtube_link:
        try:
            if "vector_store_created" not in st.session_state:
                create_vector_store(youtube_link)
                st.session_state.vector_store_created = True
                embeddings = OllamaEmbeddings(model='nomic-embed-text')
                db = "new_db_"+youtube_link.split('=')[1]
                st.session_state.retriever = Chroma(persist_directory=f"./{db}", embedding_function=embeddings).as_retriever()

            prompt_template = """You are an assistant for question-answering on a given video transcript.
            Use the following pieces of retrieved context, here context refers to the transcript of the video related to the given question use this to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise.
            Question: {question}
            Context: {context}
            Answer:
            """
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            chain_type_kwargs = {"prompt": PROMPT}
            # Accept user questions/query
            query = st.text_input("Ask questions related to the content of the video")
            if query:
                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    chain_type="stuff",
                    memory=ConversationSummaryMemory(llm=llm, memory_key='chat_history', input_key='question', output_key='answer', return_messages=True),
                    retriever=st.session_state.retriever,
                    return_source_documents=False,
                    combine_docs_chain_kwargs=chain_type_kwargs)

                response = chain.invoke(query)
                st.write(response['answer'])
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
    
    
    
# ROUGE scores for summary quality

# F1 score or accuracy for topic detection

# Exact Match, F1, or BLEU scores for question-answering performance
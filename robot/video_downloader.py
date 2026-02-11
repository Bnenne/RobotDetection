from pytubefix import YouTube
import requests, os
from dotenv import load_dotenv

load_dotenv()

api_endpoint = "https://www.thebluealliance.com/api/v3"
api_key = os.environ.get("TBA_API_KEY")

headers = {
    "X-TBA-Auth-Key": api_key
}

event_key = "2025mokc"
api_path = f"/event/{event_key}/matches"

response = requests.get(api_endpoint + api_path, headers=headers)
data = response.json()

video_keys = []

for match in data:
    if match["comp_level"] == "sf" or match["comp_level"] == "f":
        for video in match["videos"]:
            if video["type"] == "youtube":
                video_keys.append(video["key"])

download_dir = "../videos"

for key in video_keys:
    try:
        print("https://www.youtube.com/watch?v=" + key)
        yt = YouTube("https://www.youtube.com/watch?v=" + key)
        print(f"Downloading video: {yt.title}")

        video_stream = yt.streams.filter(progressive=False, file_extension='mp4').order_by('resolution').desc().first()

        video_stream.download(output_path=download_dir)
        print("Download complete!")

    except Exception as e:
        print(f"An error occurred: {e}")

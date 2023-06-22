from pytube import YouTube
from IPython.display import display,HTML
import os 

os.makedirs("./data", exist_ok=True)
url = "https://www.youtube.com/watch?v=ILqJOHYYlkc"
yt = YouTube(url)
stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
stream.download('./data/')
# HTML("""
# <video width="640" height="480" controls>
#     <source src="./홈페이지 배경 샘플 영상 - 바다.mp4" type="video/mp4">
# </video>
# """)
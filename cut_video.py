from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

start = 60
# ffmpeg_extract_subclip("data/video/ch01_00000000049000100.mp4", start, start+10,
#                        targetname="data/video/tran_dai_nghia_c9_10s.mp4")

ffmpeg_extract_subclip("data/video/ch01_00000000000000000_R.mp4", start, start+10,
                       targetname="data/video/tran_dai_nghia_c9_10s_R.mp4")
import gradio as gr
from refacer import Refacer
import argparse
import multiprocessing as mp
import os

parser = argparse.ArgumentParser(description='Refacer')
parser.add_argument("--max_num_faces", help="Max number of faces on UI", type=int, default=8)
parser.add_argument("--force_cpu", help="Force CPU mode", default=False,action="store_true")
parser.add_argument("--share_gradio", help="Share Gradio", default=True,action="store_true")
parser.add_argument("--autolaunch", help="Auto start in browser", default=False,action="store_true")
parser.add_argument("--server_name", help="Server IP address", default="127.0.0.1")
parser.add_argument("--server_port", help="Server port", type=int, default=7860)
parser.add_argument("--tensorrt", help="TensorRT activate", default=True,action="store_true")
parser.add_argument("--gpu-threads", help="number of threads to be use for the GPU", dest="gpu_threads", type=int, default=10)
parser.add_argument('--max-memory', help='maximum amount of RAM in GB to be used', dest='max_memory', type=int, default=10000)
parser.add_argument('--video_quality', help='настроить качество выходного видео', dest='video_quality', type=int, default=30, choices=range(52), metavar='[0-51]')
#parser.add_argument("--mem", help="Max memory", dest="cpu_threads", type=int, default=4)

args = parser.parse_args()

refacer = Refacer(force_cpu=args.force_cpu,tensorrt=args.tensorrt,gpu_threads=args.gpu_threads,max_memory=args.max_memory,video_quality=args.video_quality)

num_faces=args.max_num_faces

def run(*vars):
    video_path=vars[0]
    origins=vars[1:(num_faces+1)]
    destinations=vars[(num_faces+1):(num_faces*2)+1]
    thresholds=vars[(num_faces*2)+1:]
    upscaler=vars[-1]
    
    faces = []
    for k in range(0,num_faces):
        if origins[k] is not None and destinations[k] is not None:
            faces.append({
                'origin':origins[k],
                'destination':destinations[k],
                'threshold':thresholds[k]
            })

    # Преобразование upscaler в строку
    return refacer.reface(video_path,faces,str(upscaler))

origin = []
destination = []
thresholds = []
upscaler = []
upscaler = []
upscaler_models = ['None']
upscaler_models += [file for file in os.listdir('upscaler_models') if file.endswith('.onnx')]
print(upscaler_models)

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# Refacer | Repack от https://www.youtube.com/@ba1yya")
    with gr.Row():
        video=gr.Video(label="Оригинальное видео",format="mp4")
        video2=gr.Video(label="Обработанное видео",interactive=False,format="mp4")

    for i in range(0,num_faces):
        with gr.Tab(f"Лицо #{i+1}"):
            with gr.Row():
                origin.append(gr.Image(label="Лицо, которое заменяем"))
                destination.append(gr.Image(label="Лицо, на которое надо заменить"))
            with gr.Row():
                thresholds.append(gr.Slider(label="Порог",minimum=0.0,maximum=1.0,value=0.2))
    with gr.Row():
        #upscaler.append(gr.Radio(label="Upscaler", choices=models_ESRGAN, value=models_ESRGAN[0], interactive=True))
        upscaler.append(gr.Dropdown(label="Выберите модель апскейлера", choices=upscaler_models, value=upscaler_models[0], interactive=True))
    with gr.Row():
        button=gr.Button("Начать обработку", variant="primary")

    button.click(fn=run,inputs=[video]+origin+destination+thresholds+upscaler,outputs=[video2])
    
#demo.launch(share=True,server_name="0.0.0.0", show_error=True)
demo.queue().launch(show_error=True,share=args.share_gradio,server_name=args.server_name,inbrowser=args.autolaunch)

#demo.launch(share=True,server_name="0.0.0.0", show_error=True)
#demo.queue().launch(show_error=True,share=False,inbrowser=True)
#e().launch(show_error=True,share=False,inbrowser=True)
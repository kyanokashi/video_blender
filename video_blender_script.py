import cv2
import numpy as np
import threading
import pandas as pd
from multiprocessing import Pool
import subprocess as sp
from os import remove
import multiprocessing
from itertools import product 

def soft_light_blending(background, foreground):
	return np.where(foreground < 0.5, 
			                 2 * foreground * background + np.square(background) * (1 - 2*foreground), 
			                 2 * background * (1 - foreground) + np.sqrt(background) * (2 * foreground - 1))

def multiply_blending(background, foreground):
	return background * foreground

global num_videos
num_videos = 2

layer1 = list(range(1,61))
layer2 = list(range(1,61))
layer3 = list(range(1,21))

layers = [layer1, layer2, layer3]

combinations = set(product(*layers))

i = 0

path_layer1 = './BLACK_FINAL_TEST/LAYER1/BLACK_L1_'
path_layer2 = './BLACK_FINAL_TEST/LAYER2/BLACK_L2_'
path_layer3 = './BLACK_FINAL_TEST/LAYER3/BLACK_L3_'

while i < 1:
	indices = combinations.pop()
	layers = list(map(lambda x: cv2.VideoCapture(f'{path}{x}.mp4'), combinations.pop()))


class videoBlendThread(threading.Thread):
	def __init__(self, threadId, lock, frame_start):
		threading.Thread.__init__(self)
		self.threadId = threadId
		self.lock = lock
		self.frame_start = frame_start

	def run(self):
		print('Starting blending thread'+ str(self.threadId) + '\n')
		blend_layers(self.lock, self.frame_start, self.threadId)
		print('Exiting blending thread'+ str(self.threadId) + '\n')

def blend_layers(lock, frame_start, num_thread): 
	fps =  layers[0].get(cv2.CAP_PROP_FPS)

	width = layers[0].get(cv2.CAP_PROP_FRAME_WIDTH)
	height = layers[0].get(cv2.CAP_PROP_FRAME_HEIGHT)

	size = (int(width), int(height))
	video_out = cv2.VideoWriter('output{}.mp4'.format(num_thread), cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True)
	i = 0
	while i < 40:
		lock.acquire()
		for layer in layers:
			layer.set(cv2.CAP_PROP_FRAME_COUNT, i)

		ret, background = layers[0].read()
		_, middleground = layers[1].read()
		_, foreground = layers[2].read()

		lock.release()

		if ret:
			middleground = middleground.astype(float) / 255
			background = background.astype(float) / 255
			foreground = foreground.astype(float) / 255

			#soft light blending
			frame = np.where(middleground < 0.5, 
			                 2 * middleground * background + np.square(background) * (1 - 2*middleground), 
			                 2 * background * (1 - middleground) + np.sqrt(background) * (2 * middleground - 1))

			final_frame = frame * foreground

			print(frame_start, i)
			
			video_out.write((frame*255).astype(np.uint8))

			i+=1
		else:
			video_out.release()
			break


layers = [cv2.VideoCapture('./python_blend_TEST/layer_01.mp4'), 
cv2.VideoCapture('./python_blend_TEST/layer_02_SOFTLIGHT.mp4'), 
cv2.VideoCapture('./python_blend_TEST/layer_03_MULTIPLY.mp4')]

print('cpus', multiprocessing.cpu_count())
lock = threading.Lock()

fps =  layers[0].get(cv2.CAP_PROP_FPS)

total_frames = layers[0].get(cv2.CAP_PROP_FRAME_COUNT)

width = layers[0].get(cv2.CAP_PROP_FRAME_WIDTH)
height = layers[0].get(cv2.CAP_PROP_FRAME_HEIGHT)

size = (int(width), int(height))


#buffer3 = []
blend_thread0 = videoBlendThread(0, lock, 0)
blend_thread1 = videoBlendThread(1, lock, 40)
blend_thread2 = videoBlendThread(2, lock, 80)
#blend_thread3 = videoBlendThread(3, lock, 60, buffer3)

blend_thread0.start()
blend_thread1.start()
blend_thread2.start()

blend_thread0.join()
blend_thread1.join()
blend_thread2.join()

for layer in layers:
	layer.release()


list_of_output_files = ["output{}.mp4".format(i) for i in range(3)]

with open("list_of_output_files.txt", "w") as f:
        for t in list_of_output_files:
            f.write("file {} \n".format(t))

ffmpeg_cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i list_of_output_files.txt -vcodec copy " + "output.mp4"
sp.Popen(ffmpeg_cmd, shell=True).wait()

# Remove the temperory output files
for f in list_of_output_files:
	remove('{}'.format(f))
remove("list_of_output_files.txt")


print('Finished all threads \n')

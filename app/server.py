import os
import sys
import cv2
import math
import torch
import typing
import aiohttp
import asyncio
import uvicorn
import argparse
import numpy as np
import torch.nn as nn
from model import BiSeNet
from io import BytesIO
from test import evaluate
from pathlib import Path
from PIL.ExifTags import TAGS
from skimage.filters import gaussian
import torchvision.transforms as transforms
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
from starlette.background import BackgroundTask
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse

# export_file_url  = 'https://live.staticflickr.com/65535/49845513266_92f41da548_o_d.png'
# export_file_name = 'black-mask.png'

path = Path(__file__).parent


app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
	if dest.exists(): return
	async with aiohttp.ClientSession() as session:
		async with session.get(url) as response:
			data = await response.read()
			with open(dest, 'wb') as f:
				f.write(data)


async def setup_mask():
	await download_file(export_file_url, path / export_file_name)
	BLACK_IMAGE_PATH = path/export_file_name
	return BLACK_IMAGE_PATH
	
# loop = asyncio.get_event_loop()
# tasks = [asyncio.ensure_future(setup_mask())]
# BLACK_IMAGE_PATH = loop.run_until_complete(asyncio.gather(*tasks))[0]
# loop.close()

@app.route('/')
async def homepage(request):
	html_file = path / 'view' / 'index.html'
	if os.path.exists("app/newmask.png"):
		os.remove("app/newmask.png")
	else:
		print ("The file does not exist")
	return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST','GET'])
async def analyze(request):
	
	img_data = await request.form()
	img_bytes = await (img_data['file'].read())
	im = Image.open(BytesIO(img_bytes))

	def exif_remover(img):
		img_exif = img.getexif()
		if img_exif:
			for key,value in img._getexif().items():
				if TAGS.get(key) == 'Orientation':
					orientation = value
					if orientation == 1:
						return img 
					if orientation == 3:
						img = img.rotate(180)
						return img
					if orientation == 6:
						img = img.rotate(270)
						return img
					if orientation == 8:
						img= img.rotate(90)
						return img
		else:
			return img


	def resizer(img,max_size):
		if img.height > max_size or img.width > max_size:
			# if width > height:
			if img.width > img.height:
				desired_width = max_size
				desired_height = img.height / (img.width/max_size)
				
			# if height > width:
			elif img.height > img.width:
				desired_height = max_size
				desired_width = img.width / (img.height/max_size)
				
			else:
				desired_height = max_size
				desired_width = max_size
				
			# convert back to integer
			desired_height = int(desired_height)
			desired_width = int(desired_width)
				
			return img.resize((desired_width, desired_height))

		else:
			return img

	def sharpen(img):
		img = img * 1.0
		gauss_out = gaussian(img, sigma=5, multichannel=True)

		alpha = 1.5
		img_out = (img - gauss_out) * alpha + img

		img_out = img_out / 255.0

		mask_1 = img_out < 0
		mask_2 = img_out > 1

		img_out = img_out * (1 - mask_1)
		img_out = img_out * (1 - mask_2) + mask_2
		img_out = np.clip(img_out, 0, 1)
		img_out = img_out * 255
		return np.array(img_out, dtype=np.uint8)


	def hair(image, parsing, part=17, color=[ 255,0,0]):
		b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
		tar_color = np.zeros_like(image)
		tar_color[:, :, 0] = b
		tar_color[:, :, 1] = g
		tar_color[:, :, 2] = r

		image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

		if part == 12 or part == 13:
			image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
		else:
			image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]

		changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

		if part == 17:
			changed = sharpen(changed)

		changed[parsing != part] = image[parsing != part]

		return changed

	def parser(image_path='./imgs/116.jpg', cp='cp/79999_iter.pth'):
		n_classes = 19
		net = BiSeNet(n_classes=n_classes)
		device = torch.device("cpu")
		net.to(device)
		net.load_state_dict(torch.load(cp,map_location='cpu'))
		net.eval()

		to_tensor = transforms.Compose([transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

		with torch.no_grad():
			# img = Image.open(image_path)
			img = to_tensor(im)
			img = torch.unsqueeze(img, 0)
			out = net(img)[0]
			parsing = out.squeeze(0).numpy().argmax(0)
			return parsing
	

	if __name__ == '__main__':

		cp = 'app/cp/79999_iter.pth'
		
		#change img size and orientation
		im = exif_remover(im)
		im = resizer(im, 400)

		#read image for operations
		image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR) 
		ori = image.copy()
		parsing = parser(im, cp)
		part = 17
		color = [0,107, 107]
		image = hair(image, parsing, part, color)
		cv2.imwrite('app/new_makeup.png',image)   
	return FileResponse('app/new_makeup.png',media_type='image/png')


@app.route("/download",methods=['POST','GET'])
async def  download(request):
	# task = BackgroundTask(rem)
	return FileResponse("app/new_makeup.png",media_type='image/png')




if __name__ == '__main__':
	if 'serve' in sys.argv:
		uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt

import os
import sys
sys.path.append("/usr/src/slt/slt/")
# import nmt
# import train
# import inference
import argparse
import time
from datetime import datetime

import numpy as np
import cv2
import pickle
import gzip

# for i3d feature extraction
import torch
import torch.nn as nn
from models.pytorch_i3d import InceptionI3d

# for text bpemb decoding
from bpemb import BPEmb

# Sign Langauge Translation model
from signjoey.prediction import test

# Data preparation hyperparameters
window = 8
stride = 2
size = (224, 224)

# Translation Greek Decoding
vocab_size = 25000
bpemb_gr = BPEmb(lang='el', vs=vocab_size)

@csrf_exempt
def web_video_upload(request):
    return render(request, "upload.html")
    

@csrf_exempt
def rest_video_upload(request):
    if request.method == "POST" and request.FILES["video_file"]:
        video_file = request.FILES["video_file"]
        request_details = request.POST['request_details']
        fs = FileSystemStorage()
        filename = fs.save('received/' + video_file.name, video_file)
        video_url = fs.url(filename)
        print("/usr/src/slt/slt" + video_url)

        # Track processing time
        received_at = str(datetime.now())
        t_start = time.time()

        # Load model
        i3d = InceptionI3d(2000, in_channels=3)
        i3d.load_state_dict(torch.load("/usr/src/slt/slt/models/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt"))
        i3d.cuda()
        i3d = nn.DataParallel(i3d)
        i3d.eval()

        # Data preparation
        # Video to Frames 4D matrix (F x C x W x W)
        vidcap = cv2.VideoCapture("/usr/src/slt" + video_url)
        success, image = vidcap.read()
        count = 0
        processed_images = []
        print('before', success)
        while success:
            count += 1
            success, image = vidcap.read()
            if image is None:
                print('ok')
                break
            image_resized = cv2.resize(image, size)
            processed_images.append(image_resized)
        processed_images = np.transpose(processed_images, (0, 3, 1, 2))
        processed_images = (processed_images / 255.0) * 2 - 1
        print(processed_images.shape)

        # Frames matrix (4D) to batch of window frames matrix with sliding window (5D)
        frames = processed_images.shape[0]
        batch_processed_images = []
        for j in range(0, frames - window + 1, stride):
            batch_processed_images.append(processed_images[j:j+window])
        batch_processed_images = np.array(batch_processed_images)
        batch_processed_images = np.transpose(batch_processed_images, (0, 2, 1, 3, 4))
        print(batch_processed_images.shape)

        # Extract features with pretrained i3d network
        with torch.no_grad():
            batch_processed_images = i3d.forward(torch.tensor(batch_processed_images, dtype=torch.float))
            print(batch_processed_images.shape)
            batch_processed_images = batch_processed_images.reshape((-1, 1024))
        print(batch_processed_images.shape)

        # Store data for inference
        data = {"name": video_url,
                "signer": "random",
                "gloss": "random",
                "text": "random",
                "sign": batch_processed_images.cpu()}
        with gzip.open("/usr/src/slt/slt/data/inference.i3d", 'wb') as handle:
            pickle.dump([data], handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.remove("/usr/src/slt" + video_url)

        # Track processing time
        t_pre_processed = time.time()

        # Perform sign language translation
        test(cfg_file="/usr/src/slt/slt/models/sign_1_1_8_2.yaml",
             ckpt="/usr/src/slt/slt/models/s2t_lwta_1_1_8_2_meleti_ab_bpe_25000/best.ckpt",
             output_path="/usr/src/slt/slt/results/")
        # Read result
        with open('/usr/src/slt/slt/results/.BW_01.A_1.test.txt', encoding="utf-8") as f:
            first_line = f.readline()
        first_line = first_line.split("|")[1].replace(" ", "").replace("▁", " ")[1:]
        print(first_line)

        t_inferred = time.time()

        pre_process_duration = round(t_pre_processed - t_start, 2)
        inference_duration = round(t_inferred - t_pre_processed, 2)

        return JsonResponse({
            "video_stored_at": video_url,
            "request_details": {"received_at": received_at,
                                "split_duration": pre_process_duration,
                                "inference_duration": inference_duration,
                                "other": request_details},
            "sign_language_translation": first_line,
        }, json_dumps_params={'ensure_ascii': False})

    return JsonResponse({'API_type': 'GET'})


def break_to_images(video_url, size=(227, 227)):
    video_path = "/usr/src/slt" + video_url
    if not os.path.isdir(video_path[:-4]):
        os.mkdir(video_path[:-4])

    vidcap = cv2.VideoCapture(video_path)
    # if not os.path.isdir(filepath):
    #     os.mkdir(filepath)

    success, image = vidcap.read()
    count = 0

    print('before', success)
    while success:
        count += 1
        success, image = vidcap.read()
        if image is None:
            print('ok')
            break
        # image_crop = image[:, 280:-280]
        image_resized = cv2.resize(image, size)
        #### Save Resized Frames
        cv2.imwrite(video_path[:-4] + '/images' + str(count).zfill(4) + '.png', image_resized)

    return count



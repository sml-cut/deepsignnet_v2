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



@csrf_exempt
def web_video_upload(request):
    if request.method == "POST" and request.FILES["video_file"]:
        test=2
        video_file = request.FILES["video_file"]
        fs = FileSystemStorage()
        filename = fs.save('received/' + video_file.name, video_file)
        video_url = fs.url(filename)
        print(video_url)
        return render(request, "upload.html", {
            "video_url": video_url,
            "video_translation": "This is a dummy video translation"
        })
    return render(request, "upload.html")
    

@csrf_exempt
def rest_video_upload(request):
    if request.method == "POST" and request.FILES["video_file"]:
        video_file = request.FILES["video_file"]
        request_details = request.POST['request_details']
        fs = FileSystemStorage()
        filename = fs.save('received/' + video_file.name, video_file)
        video_url = fs.url(filename)
        print(video_url)
        with open('/usr/src/slt/slt/video/video_location.sign', 'w') as out_file:
            # out_file.write("/usr/src/slt" + video_url[:-4] + "/")
            out_file.write("/usr/src/slt/slt/video/april/")

        received_at = str(datetime.now())
        t_start = time.time()

        frames = break_to_images(video_url)
        t_split = time.time()

        # Run inference
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        #nmt_parser = argparse.ArgumentParser()
        #nmt.add_arguments(nmt_parser)
        #FLAGS, unparsed = nmt_parser.parse_known_args()
        #default_hparams = nmt.create_hparams(FLAGS)
        #train_fn = train.train
        #inference_fn = inference.inference
        #nmt.run_main(FLAGS, default_hparams, train_fn, inference_fn)
        time.sleep(1)

        # Read result
        with open('/usr/src/slt/slt/video/predictions.de') as f:
            first_line = f.readline()

        t_inferred = time.time()

        split_duration = round(t_split - t_start, 2)
        inference_duration = round(t_inferred - t_split, 2)


        return JsonResponse({
            "video_stored_at": video_url,
            "request_details": {"received_at": received_at,
                                "split_duration": split_duration,
                                "inference_duration": inference_duration,
                                "frames": frames,
                                "other": request_details},
            "sign_language_translation": first_line,
        })
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



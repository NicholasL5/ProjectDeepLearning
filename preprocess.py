import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
import imageio


def video2frame(filepath, resize=(224,224)):

    cap = cv2.VideoCapture(filepath)
    # num of frames
    len_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(len_frames)
    len_frames = int(len_frames)
    frames = []
    try:
        for i in range(len_frames):
            _, frame = cap.read()
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224, 224, 3))
            frames.append(frame)
    except:
        print("error: ", filepath, len_frames, i)
    finally:
        frames = np.array(frames)
        cap.release()

    # flows = calc_optical_flow(frames)
    #
    # result = np.zeros((len(flows), 224, 224, 5))
    # result[..., :3] = frames
    # result[..., 3:] = flows
    #
    # frames = uniform_sampling(result, 20)
    return frames


def frame_difference(frames):
    num_frames = len(frames)
    out = []
    for i in range(num_frames - 1):
        out.append(cv2.subtract(frames[i + 1], frames[i]))

    return np.array(out)


def uniform_sampling(frames, target_frames=10):
    num_frames = len(frames)
    skip_frames = num_frames//target_frames
    out = []

    for i in range(target_frames):
        out.append(frames[i * skip_frames])

    return np.array(out)


def flow_to_color(video):
    rgb_flows = []

    cap = cv2.VideoCapture(video)

    ret, first_frame = cap.read()

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculates dense optical flow by Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                           None,
                                           0.5, 3, 15, 3, 5, 1.2, 0)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        resized_rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        rgb_flows.append(resized_rgb)

        prev_gray = gray

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    # rgb_flows.append(np.zeros((224, 224, 3)))
    # cv2.destroyAllWindows()

    return np.array(rgb_flows)



def Save2Npy(file_dir, save_dir):
    """Transfer all the videos and save them into specified directory
    Args:
        file_dir: source folder of target videos
        save_dir: destination folder of output .npy files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # List the files
    videos = os.listdir(file_dir)
    for v in tqdm(videos):
        # Split video name
        video_name = v.split('.')[0]
        # Get src
        video_path = os.path.join(file_dir, v)
        # Get dest
        save_path = os.path.join(save_dir, video_name + '.npy')
        # Load and preprocess video
        data = video2frame(video_path, resize=(224, 224))
        data = np.uint8(data)
        # Save as .npy file
        np.save(save_path, data)

    return None

def Save2Npy2(file_dir, save_dir):
    """Transfer all the videos and save them into specified directory
    Args:
        file_dir: source folder of target videos
        save_dir: destination folder of output .npy files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # List the files
    videos = os.listdir(file_dir)
    for v in tqdm(videos):
        # Split video name
        video_name = v.split('.')[0]
        # Get src
        video_path = os.path.join(file_dir, v)
        # Get dest
        save_path = os.path.join(save_dir, video_name + '.npy')
        # Load and preprocess video
        data = flow_to_color(video_path)
        data = np.uint8(data)
        # Save as .npy file
        np.save(save_path, data)

    return None

def calc_optical_flow(frames):
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]

    dense_flow = []

    for i in range(1, len(gray_frames)):
        prev_frame = gray_frames[i-1]
        cur_frame = gray_frames[i]

        flow = cv2.calcOpticalFlowFarneback(prev_frame, cur_frame, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # normalize
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)

        dense_flow.append(flow)
    # Padding the last frame as empty array
    dense_flow.append(np.zeros((224, 224, 2)))

    return np.array(dense_flow, dtype=np.float32)


if __name__ == "__main__":
    test_frames = video2frame(r"E:\Projects\DeepLearning\RWF-2000\train\Fight\0_DzLlklZa0_3.avi")
    # print(test_frames.shape)
    # frame = test_frames[11]
    #
    # # Select only the first three channels
    # frame_first_3_channels = frame[..., :3]
    #
    # # Display the frame with only the first 3 channels
    # plt.imshow(frame_first_3_channels.astype(np.uint8))
    # plt.show()

    skipped_frame = uniform_sampling(test_frames)
    print(skipped_frame.shape)

    diff = frame_difference(skipped_frame)
    for frame in diff:
        plt.imshow(frame.astype(np.uint8))
        plt.show()

    # optical_flow = calc_optical_flow(test_frames)
    # print(optical_flow.shape)

    # rgb_opt = flow_to_color(r"E:\Projects\DeepLearning\RWF-2000\train\Fight\2sy_eLpH-PI_0.avi")
    # print(rgb_opt.shape)
    # print(rgb_opt)


    # frames = uniform_sampling(rgb_opt, 20)
    # for frame in frames:
    #     plt.imshow(frame.astype(np.uint8))
    #     plt.show()

    # for rgb in rgb_opt:
    #     cv2.imshow("test", rgb)
    #     cv2.waitKey(0)  # Wait indefinitely for a key press
    #     cv2.destroyAllWindows()  # Close the window when a key is pressed
    #
    # source_path = r'E:\Peliculas'
    # target_path = r'E:\movie_fights'
    #
    #
    # for f2 in ['fights', 'noFights']:
    #     path1 = os.path.join(source_path, f2)
    #     path2 = os.path.join(target_path, f2)
    #     Save2Npy(file_dir=path1, save_dir=path2)

from collections import deque
from datetime import datetime
import cv2
import numpy as np
import tqdm
from scipy.ndimage.measurements import label
import torch
import clip
from PIL import Image


def slidingWindow(image_size, init_size=(96,96), x_overlap=0.5, y_step=0.05,
                  x_range=(0, 1), y_range=(0, 1), scale=0):
    windows = []
    h, w = image_size[1], image_size[0]
    for y in range(int(y_range[0] * h), int(y_range[1] * h), int(y_step * h)):
        win_width = int(init_size[0] + (scale * (y - (y_range[0] * h))))
        win_height = int(init_size[1] + (scale * (y - (y_range[0] * h))))
        if y + win_height > int(y_range[1] * h) or win_width > w:
            break
        x_step = int((1 - x_overlap) * win_width)
        for x in range(int(x_range[0] * w), int(x_range[1] * w), x_step):
            windows.append((x, y, x + win_width, y + win_height))
    return windows

class Detector:
    def __init__(self, 
                 init_size=(64, 64), 
                 x_overlap=0.5, 
                 y_step=0.05,
                 x_range=(0, 1), 
                 y_range=(0, 1), 
                 scale=1.5):
        self.init_size = init_size
        self.x_overlap = x_overlap
        self.y_step = y_step
        self.x_range = x_range
        self.y_range = y_range
        self.scale = scale
        self.windows = None

    def load_clip_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)
        self.car_text = "car"
        self.non_car_texts = [
            "sky",
            "road",
            "road fence",
            "road sign",
            "tree",
            "grass",
            "building",
            "person",
            "cloud",
            "leaf",
            "blacktop" # 沥青路
        ]


    def classify_with_clip(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_pil = Image.fromarray(image)
        positive_windows = []
        for (x_upper, y_upper, x_lower, y_lower) in self.windows:
            window = image_pil.crop((x_upper, y_upper, x_lower, y_lower))
            window_input = self.preprocess(window).unsqueeze(0).to(self.device)
            with torch.no_grad():
                window_features = self.model.encode_image(window_input)
            all_texts = [self.car_text] + self.non_car_texts
            text_features = self.model.encode_text(clip.tokenize(all_texts).to(self.device))
            similarity = (window_features @ text_features.T).squeeze().tolist()
            car_similarity = similarity[0]
            non_car_similarities = similarity[1:]
            if car_similarity > max(non_car_similarities) + 1.0 and car_similarity > 33:
                # print(f"汽车相似度: {car_similarity}")
                positive_windows.append((x_upper, y_upper, x_lower, y_lower))
        return positive_windows



    def detectVideo(self, video_capture=None, num_frames=9, threshold=80,
                    min_bbox=None, show_video=False, draw_heatmap=True,
                    draw_heatmap_size=0.2, write=True, write_fps=24):
        cap = video_capture
        if not cap.isOpened():
            raise RuntimeError("Error opening VideoCapture.")
        fps_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        (grabbed, frame) = cap.read()
        (h, w) = frame.shape[:2]        
        self.windows = slidingWindow((w, h), 
                                init_size=self.init_size,
                                x_overlap=self.x_overlap, 
                                y_step=self.y_step,
                                x_range=self.x_range, 
                                y_range=self.y_range, 
                                scale=self.scale)
        self.threshold = threshold
        self.load_clip_model()
        if min_bbox is None:
            min_bbox = (int(0.02 * w), int(0.02 * h))
        inset_size = (int(draw_heatmap_size * w), int(draw_heatmap_size * h))
        if write:
            vidFilename = datetime.now().strftime("%Y%m%d%H%M") + ".avi"
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            writer = cv2.VideoWriter(vidFilename, fourcc, write_fps, (w, h))

        current_heatmap = np.zeros((frame.shape[:2]), dtype=np.uint8)
        summed_heatmap = np.zeros_like(current_heatmap, dtype=np.uint8)
        last_N_frames = deque(maxlen=num_frames)
        heatmap_labels = np.zeros_like(current_heatmap, dtype=int)
        weights = np.linspace(1 / (num_frames + 1), 1, num_frames)
        bar = tqdm.tqdm(total=fps_total, desc="Detect process")
        while True:
            (grabbed, frame) = cap.read()
            bar.update()
            if not grabbed:
                bar.close()
                break
            current_heatmap[:] = 0
            summed_heatmap[:] = 0

            for (x_upper, y_upper, x_lower, y_lower) in self.classify_with_clip(frame):
                current_heatmap[y_upper:y_lower, x_upper:x_lower] += 5

            last_N_frames.append(current_heatmap)
            for i, heatmap in enumerate(last_N_frames):
                cv2.add(summed_heatmap, (weights[i] * heatmap).astype(np.uint8),
                        dst=summed_heatmap)

            cv2.dilate(summed_heatmap, np.ones((7, 7), dtype=np.uint8),
                       dst=summed_heatmap)

            if draw_heatmap:
                inset = cv2.resize(summed_heatmap, inset_size,
                                   interpolation=cv2.INTER_AREA)
                inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
                frame[:inset_size[1], :inset_size[0], :] = inset

            summed_heatmap[summed_heatmap <= threshold] = 0

            num_objects = label(summed_heatmap, output=heatmap_labels)

            for obj in range(1, num_objects + 1):
                (Y_coords, X_coords) = np.nonzero(heatmap_labels == obj)
                x_upper, y_upper = min(X_coords), min(Y_coords)
                x_lower, y_lower = max(X_coords), max(Y_coords)

                if (x_lower - x_upper > min_bbox[0]
                        and y_lower - y_upper > min_bbox[1]):
                    cv2.rectangle(frame, (x_upper, y_upper), (x_lower, y_lower),
                                  (0, 255, 0), 6)
            if write:
                writer.write(frame)
            if show_video:
                cv2.imshow("Detection", frame)
                cv2.waitKey(1)
        cap.release()
        if write:
            writer.release()

if __name__ == "__main__":
    detector = Detector(
        init_size=(100,80), 
        x_overlap=0.75, 
        y_step=0.01,
        x_range=(0.02, 0.98), 
        y_range=(0.55, 0.89), 
        scale=1.05
        )
    video_file = "videos/test_video.mp4"   
    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(
        video_capture=cap,
        draw_heatmap_size=0.2,
        threshold=80 
        )


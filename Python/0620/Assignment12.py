import matplotlib.pyplot as plt
import librosa
import librosa.display
import os 
import glob
import random
import numpy as np
import IPython
from PIL import Image

class Sound_Augmentation:
    def __init__(self, input_path):
        self.input_path = input_path

    def sound_augmentation(self, input_path):
        # Read files' directory and name
        dir, file = os.path.split(input_path)
        folder_name = dir.rsplit('/')[2]
        name = os.path.basename(input_path).rsplit('.wav')[0]
        
        # make image_extraction_data folder
        os.makedirs("./image_extraction_data", exist_ok=True)
        os.makedirs("./image_extraction_data/MelSepctrogram", exist_ok=True)
        os.makedirs("./image_extraction_data/STFT", exist_ok=True)
        os.makedirs("./image_extraction_data/waveshow", exist_ok=True)
        for (path, dir, files) in os.walk('./raw_data/'):
            for d in dir :
                os.makedirs(f"./image_extraction_data/MelSepctrogram/{d}", exist_ok=True)
                os.makedirs(f"./image_extraction_data/STFT/{d}", exist_ok=True)
                os.makedirs(f"./image_extraction_data/waveshow/{d}", exist_ok=True)
        
        # MelSepctrogram
        data, sr = librosa.load(input_path, sr=22050)
        stft = librosa.stft(data)
        mel_spec = librosa.feature.melspectrogram(S=abs(stft))
        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        
        # 0 ~ 10 sec 
        start_time = 0 
        end_time = 10
        start_sample = sr * start_time
        end_sample = sr * end_time
        data_selection_stft = data[start_sample : end_sample]

        stft_temp = librosa.stft(data_selection_stft)
        mel_spec_temp = librosa.feature.melspectrogram(S=abs(stft_temp))
        mel_spec_db_temp = librosa.amplitude_to_db(mel_spec_temp, ref=np.max)
        plt.figure(figsize=(12,4))
        librosa.display.specshow(mel_spec_db_temp, sr=sr, x_axis='time', y_axis='hz')
        plt.axis('off')
        plt.savefig(f"./image_extraction_data/MelSepctrogram/{folder_name}/{name}_mel_spec_0-10.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"{file} / MelSepctrogram - 0 to 10 secs conversion completed!")

        # noise
        stft = librosa.stft(data_selection_stft)
        mel_spec = librosa.feature.melspectrogram(S=abs(stft))
        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        noise = 0.005 * np.random.randn(*mel_spec_db.shape)
        augmented_spec = mel_spec_db + noise
        augmented_spec_db = librosa.amplitude_to_db(augmented_spec, ref=np.max)
        plt.figure(figsize=(12,4))
        librosa.display.specshow(augmented_spec_db, sr=sr, x_axis='time', y_axis='hz')
        plt.axis('off')
        plt.savefig(f"./image_extraction_data/MelSepctrogram/{folder_name}/{name}_mel_spec_0-10_aug_noise.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"{file} / MelSepctrogram - Noise Augmentation completed!")
        
        # stretch
        rate = np.random.uniform(low=0.8, high=1.2)
        stretched = librosa.effects.time_stretch(data, rate=rate)
        stft_stretched = librosa.stft(stretched)
        mel_spec_stretched = librosa.feature.melspectrogram(S=abs(stft_stretched))
        stretched_stft_db = librosa.amplitude_to_db(mel_spec_stretched, ref=np.max)
        plt.figure(figsize=(12,4))
        librosa.display.specshow(stretched_stft_db, sr=sr, x_axis='time', y_axis='hz')
        plt.axis('off')
        plt.savefig(f"./image_extraction_data/MelSepctrogram/{folder_name}/{name}_mel_spec_0-10_aug_stretch.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"{file} / MelSepctrogram - Stretch Augmentation completed!")
        
        # STFT
        data, sr = librosa.load(input_path, sr=22050)
        stft = librosa.stft(data)
        stft_db = librosa.amplitude_to_db(abs(stft))
        
        # 0초 ~ 10초 
        start_time = 0 
        end_time = 10
        start_sample = sr * start_time
        end_sample = sr * end_time
        data_selection_stft = data[start_sample : end_sample]
        stft_temp = librosa.stft(data_selection_stft)
        stft_db_temp = librosa.amplitude_to_db(abs(stft_temp))
        plt.figure(figsize=(12,4))
        librosa.display.specshow(stft_db_temp, sr=sr, x_axis='time', y_axis='hz')
        plt.axis('off')
        plt.savefig(f"./image_extraction_data/STFT/{folder_name}/{name}_STFT_0-10.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"{file} / STFT - 0 to 10 secs conversion completed!")

        # noise
        noise = 0.005 * np.random.randn(*data_selection_stft.shape)
        augmented_data_section = data_selection_stft + noise
        augmented_stft = librosa.stft(augmented_data_section)
        augmented_stft_db = librosa.amplitude_to_db(abs(augmented_stft))
        plt.figure(figsize=(12,4))
        librosa.display.specshow(augmented_stft_db, sr=sr, x_axis='time', y_axis='hz')
        plt.axis('off')
        plt.savefig(f"./image_extraction_data/STFT/{folder_name}/{name}_STFT_0-10_aug_noise.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"{file} / STFT - Noise Augmentation completed!")
        
        # stretch
        rate = 0.8 + np.random.random() * 0.4
        stretched_data_section = librosa.effects.time_stretch(data_selection_stft, rate=rate)
        stretched_stft = librosa.stft(stretched_data_section)
        stretched_stft_db = librosa.amplitude_to_db(abs(stretched_stft))
        plt.figure(figsize=(12,4))
        librosa.display.specshow(stretched_stft_db, sr=sr, x_axis='time', y_axis='hz')
        plt.axis('off')
        plt.savefig(f"./image_extraction_data/STFT/{folder_name}/{name}_STFT_0-10_aug_stretch.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"{file} / STFT - Stretch Augmentation completed!")
        
        # Waveshow
        data, sr = librosa.load(input_path, sr=22050)    

        # 0초 ~ 10초 
        start_time = 0
        end_time = 10
        start_sample = sr * start_time
        end_sample = sr * end_time
        data_section = data[start_sample:end_sample]
        plt.figure(figsize=(12,4))
        librosa.display.waveshow(data_section, color='purple')
        plt.axis('off')
        plt.savefig(f"./image_extraction_data/waveshow/{folder_name}/{name}_waveshow_0-10.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"{file} / Waveshow - 0 to 10 secs conversion completed!")

        # noise
        # 노이즈 추가
        noise = 0.05 * np.random.rand(*data_section.shape)
        data_noise = data_section + noise
        plt.figure(figsize=(12,4))
        librosa.display.waveshow(data_noise, color='purple')
        plt.axis('off')
        plt.savefig(f"./image_extraction_data/waveshow/{folder_name}/{name}_waveshow_0-10_aug_noise.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"{file} / Waveshow - Noise Augmentation completed!")

        # stretch
        data_stretch = librosa.effects.time_stretch(data_section, rate=0.8)
        plt.figure(figsize=(12,4))
        librosa.display.waveshow(data_stretch, color='purple')
        plt.axis('off')
        plt.savefig(f"./image_extraction_data/waveshow/{folder_name}/{name}_waveshow_0-10_aug_stretch.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"{file} / Waveshow - Stretch Augmentation completed!")

    def expend2square(self, pil_image, background_color):
        width, height = pil_image.size
        if width == height:
            return pil_image
        elif width > height :
            result = Image.new(pil_image.mode, (width, width), background_color)
            result.paste(pil_image, (0, (width-height) // 2))
            return result
        else:
            result = Image.new(pil_image.mode, (height, height), background_color)
            result.paste(pil_image, ((height-width) // 2, 0))
            return result 

    def resize_with_padding(self, pil_image, new_size, background_color):
        img = self.expend2square(pil_image, background_color)
        img = img.resize((new_size[0], new_size[1]), Image.ANTIALIAS)
        return img

    def get_files_paths(self, input_path):
        file_list = []
        for (path, dir, files) in os.walk(input_path):
            for f in files:
                file_list.append(os.path.join(path, f))
        return file_list
    
if __name__ == "__main__":
    input_path = "./raw_data/"
    sg = Sound_Augmentation(input_path)
    file_list =  sg.get_files_paths(input_path)
    for item in file_list:
        sg.sound_augmentation(item)

    os.makedirs("./final_data", exist_ok=True)
    os.makedirs("./final_data/MelSepctrogram", exist_ok=True)
    os.makedirs("./final_data/STFT", exist_ok=True)
    os.makedirs("./final_data/waveshow", exist_ok=True)
    for (path, dir, files) in os.walk(input_path):
        for d in dir :
            os.makedirs(f"./final_data/MelSepctrogram/{d}", exist_ok=True)
            os.makedirs(f"./final_data/STFT/{d}", exist_ok=True)
            os.makedirs(f"./final_data/waveshow/{d}", exist_ok=True)
    img_path_list = glob.glob(os.path.join('./image_extraction_data/', "*", "*", "*.png"))

    for img_path in img_path_list:
        dir, file = os.path.split(img_path)
        dir1 = dir.rsplit('\\')[1]
        dir2 = dir.rsplit('\\')[2]
        name = os.path.basename(img_path).rsplit('.png')[0]
        img = Image.open(img_path)
        img_new = sg.resize_with_padding(img, (255,255), (0,0,0)) 
        save_file_name = f"./final_data/{dir1}/{dir2}/{name}.png"
        img_new.save(save_file_name, "png")
        print(f"./final_data/{dir1}/{dir2}/{name}.png saved!")
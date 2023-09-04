"""
gya@stu.xju.edu.cn

Add_noise - function

This file contains noise_segment, data_analysis and add_noise (single and random noise)

"""
import os
import wave
import random
import librosa
import soundfile as sf
import numpy as np
import shutil
from glob import glob
from numpy.linalg import norm

# 计算数据集中最长音频时长，便于对噪声文件切段
def count_duration():
    folder_path = "/home/GaoYA/Code/DNet-main/add_noisy/clean_audio/train"  # 文件夹路径

    max_duration = 0  # 最长音频的持续时间

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_path.endswith(".wav"):
            # 打开 WAV 文件
            with wave.open(file_path, "rb") as wav_file:
                # 获取音频持续时间（秒）
                duration = wav_file.getnframes() / wav_file.getframerate()
                # 更新最长音频的持续时间
                max_duration = max(max_duration, duration)

    # 输出最长音频的持续时间
    print("最长音频持续时间：", max_duration, "秒")

# 对噪声文件切段（由于噪声音频过长，直接处理过慢）
def noisy_segment():

    folder_path = "/home/GaoYA/Code/DANet-main/add_noisy/noisy_data/traffic_noisy"  # 文件夹路径
    segment_duration = 520  # 切段的目标时长（秒）

    # 创建保存切段音频的目录
    output_folder_path = os.path.join(folder_path, "segments")
    os.makedirs(output_folder_path, exist_ok=True)

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_path.endswith(".wav"):
            # 读取音频文件
            audio, sr = sf.read(file_path)

            # 计算音频的时长
            duration = librosa.get_duration(audio, sr=sr)

            num_segments = int(duration / segment_duration)
            remaining_duration = duration % segment_duration

            # 分段切割音频
            for i in range(num_segments):
                start = i * segment_duration
                end = (i + 1) * segment_duration
                segment = audio[int(start * sr):int(end * sr)]
                segment_file_name = f"{os.path.splitext(file_name)[0]}_{i}.wav"
                segment_file_path = os.path.join(output_folder_path, segment_file_name)
                sf.write(segment_file_path, segment, sr)

            if remaining_duration > 0:
                # 保存最后一段不足520秒的音频
                last_segment = audio[-int(remaining_duration * sr):]
                last_segment_file_name = f"{os.path.splitext(file_name)[0]}_{num_segments}.wav"
                last_segment_file_path = os.path.join(output_folder_path, last_segment_file_name)
                sf.write(last_segment_file_path, last_segment, sr)

# 对干净语音增加噪声（第一种：噪声文件唯一）
def add_single_noisy():
    # 设置文件路径和信噪比范围
    clean_dir = '/home/GaoYA/Code/DANet-main/data/wav'
    noise_file = '/home/GaoYA/Code/DANet-main/data/Traffic_05.wav'
    noisy_dir = '/home/GaoYA/Code/DANet-main/data/audio_with_traffic/'
    SNR_range = (0, 20)
    eps = 1e-8

    # 创建输出目录
    if not os.path.exists(noisy_dir):
        os.makedirs(noisy_dir)

    # 为每个干净语音文件添加加性噪声
    for filename in os.listdir(clean_dir):
        if not filename.endswith('.wav'):
            continue

        # 读取语音文件和噪声文件
        clean, fs = sf.read(os.path.join(clean_dir, filename))
        # if fs != 16000:
        #     if len(clean) == 2:
        #         clean = clean.transpose(1, 0)
        #         clean = librosa.resample(clean, fs, 16000)
        #
        #         clean = clean.transpose(1, 0)
        noise, sr = sf.read(noise_file)
        # noise = noise[:, 0]  # 针对噪声是双通道数据，需要先转换为单通道数据
        if fs != sr:
            noise = librosa.resample(noise, sr, fs)

        clean_length = clean.shape[0]
        noise_length = noise.shape[0]

        # 同步噪声和音频长度
        if noise_length > clean_length:
            st = np.random.randint(noise_length - clean_length + 1)
            noise_clip = noise[st:st+clean_length]
        else:
            # pad_length = clean_length - noise_length
            # noise_clip = np.pad(noise, (0, pad_length), mode='constant')
            num_copies = clean_length // noise_length

            # 复制短数组并拼接成与长数组长度一致的数组
            noise_clip = np.concatenate([noise] * num_copies + [noise[:clean_length % noise_length]])

        if clean.shape[-1] == 2:
            # noise_clip_out = np.column_stack((noise_clip, noise_clip))
            noise_clip_ = noise_clip.reshape(noise_clip.shape[0], 1)
            noise_clip_out = np.concatenate([noise_clip_, noise_clip_], axis = -1)
        else:
            noise_clip_out = noise_clip

        # 确定信噪比
        SNR_dB = np.random.uniform(low=SNR_range[0], high=SNR_range[1])
        new_noise = noise_clip_out / norm(noise_clip_out) * norm(clean) / (10.0 ** (0.05 * SNR_dB))
        noisy = clean + noise_clip_out

        # 归一化
        max_amp = np.max(np.abs(noisy))
        max_amp = np.maximum(max_amp, eps)
        noisy_scale = 1. / max_amp
        noisy = noisy * noisy_scale

        # # 计算噪声标准差并调整音量
        # clean_rms = np.sqrt(np.mean(clean ** 2))
        # noise_rms = np.sqrt(np.mean(noise_clip_out ** 2))
        # noise_std = clean_rms / SNR
        # noise_scaled = noise_clip_out * (noise_std / noise_rms)

        # 添加噪声并将结果写入输出目录中
        # noisy = clean + noise_scaled
        output_file = os.path.join(noisy_dir, filename)
        sf.write(output_file, noisy, fs)

# 对干净语音加噪（第二种：噪声文件不唯一），配合main函数使用
def add_noise(clean_audio, noise_audio, snr):
    # 调整噪声音量以达到目标信噪比
    clean_rms = np.sqrt(np.mean(clean_audio ** 2))
    noise_rms = np.sqrt(np.mean(noise_audio ** 2))
    noise_adjusted = noise_audio * (clean_rms / noise_rms) / (10 ** (snr / 20))

    # 重复或截取噪声以与干净语音长度相匹配
    if clean_audio.ndim == 1:
        if len(clean_audio) >= len(noise_adjusted):
            repeated_noise = np.tile(noise_adjusted, (len(clean_audio) // len(noise_adjusted)) + 1)[:len(clean_audio)]
            noisy_audio = clean_audio + repeated_noise
        else:
            start_idx = random.randint(0, len(noise_adjusted) - len(clean_audio))
            noisy_audio = clean_audio + noise_adjusted[start_idx:start_idx + len(clean_audio)]
    else:  # 双通道
        if len(clean_audio) >= len(noise_adjusted):
            repeated_noise = np.tile(noise_adjusted, ((len(clean_audio) // len(noise_adjusted)) + 1, 1))
            repeated_noise = repeated_noise[:len(clean_audio), :]
            noisy_audio = clean_audio + repeated_noise
        else:
            start_idx = random.randint(0, len(noise_adjusted) - len(clean_audio))
            noisy_audio = clean_audio + noise_adjusted[start_idx:start_idx + len(clean_audio), :]

    return noisy_audio

# 对干净语音加噪，噪声类型为babble(415条干净语音被加噪)、traffic(310条干净语音被加噪)、wind(310条干净语音被加噪)
def main():
    clean_audio_dir = "/home/GaoYA/Code/DANet-main/add_noisy/clean_audio/test"  # 干净语音文件夹路径
    wind_noisy_dir = "/home/GaoYA/Code/DANet-main/add_noisy/noisy_data/wind_noisy/segments"  # 风噪声文件夹路径
    traffic_noisy_dir = "/home/GaoYA/Code/DANet-main/add_noisy/noisy_data/traffic_noisy/segments"  # 交通噪声文件夹路径
    babble_noisy_dir = "/home/GaoYA/Code/DANet-main/add_noisy/noisy_data/babble_noisy/segments"  # 嘈杂人声文件夹路径
    output_dir = "/home/GaoYA/Code/DANet-main/add_noisy/test_with_noisy"  # 输出文件夹路径

    clean_audio_count = 33  # 1035 for train, 33 for test
    babble_noise_count = 13  # 415 for train, 13 for test
    wind_noise_count = 10  # 310 for train, 10 for test
    traffic_noise_count = 10  # 310 for train, 10 for test

    snr_babble_range = (0, 20)
    snr_wind_range = (0, 20)
    snr_traffic_range = (0, 20)

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有干净语音文件路径
    clean_audio_files = os.listdir(clean_audio_dir)

    # 随机选择嘈杂人声、风噪声和交通噪声的文件
    babble_noise_files = random.choices(os.listdir(babble_noisy_dir), k=babble_noise_count)
    wind_noise_files = random.choices(os.listdir(wind_noisy_dir), k=wind_noise_count)
    traffic_noise_files = random.choices(os.listdir(traffic_noisy_dir), k=traffic_noise_count)

    # 随机选择不重复的干净语音索引
    all_indices = list(range(clean_audio_count))
    babble_indices = random.sample(all_indices, babble_noise_count)
    all_indices = list(set(all_indices) - set(babble_indices))
    wind_indices = random.sample(all_indices, wind_noise_count)
    traffic_indices = random.sample(list(set(all_indices) - set(wind_indices)), traffic_noise_count)

    # 为嘈杂噪声的干净语音加噪
    for i in range(babble_noise_count):
        clean_index = babble_indices[i]
        clean_audio_path = os.path.join(clean_audio_dir, clean_audio_files[clean_index])
        output_path = os.path.join(output_dir,
                                   f"{os.path.splitext(clean_audio_files[clean_index])[0]}_babble_noisy.wav")
        babble_noise_path = os.path.join(babble_noisy_dir, babble_noise_files[i])

        # 读取干净语音和嘈杂噪声
        clean_audio, sr = librosa.load(clean_audio_path)
        babble_noise, sr_noisy = librosa.load(babble_noise_path)

        if sr_noisy != sr:
            babble_noise = librosa.resample(babble_noise, sr_noisy, sr)

        # 转换噪声为双通道（如果干净语音是双通道）
        if clean_audio.ndim == 2:
            if len(clean_audio) == 2:
                clean_audio = np.transpose(clean_audio)
                babble_noise = np.stack([babble_noise] * 2, axis=1)
            elif len(clean_audio) != 2:
                babble_noise = np.stack([babble_noise] * 2, axis=1)

        # 随机选择信噪比
        snr = random.uniform(*snr_babble_range)

        # 加噪
        noisy_audio = add_noise(clean_audio, babble_noise, snr)

        # 保存加噪后的音频
        sf.write(output_path, noisy_audio, sr)

    # 为风噪声的干净语音加噪
    for i in range(wind_noise_count):
        clean_index = wind_indices[i]
        clean_audio_path = os.path.join(clean_audio_dir, clean_audio_files[clean_index])
        output_path = os.path.join(output_dir, f"{os.path.splitext(clean_audio_files[clean_index])[0]}_wind_noisy.wav")
        wind_noise_path = os.path.join(wind_noisy_dir, wind_noise_files[i])

        # 读取干净语音和风噪声
        clean_audio, sr = librosa.load(clean_audio_path)
        wind_noise, sr_noisy = librosa.load(wind_noise_path)

        if sr_noisy != sr:
            wind_noise = librosa.resample(wind_noise, sr_noisy, sr)

        # 转换噪声为双通道（如果干净语音是双通道）
        if clean_audio.ndim == 2:
            if len(clean_audio) == 2:
                clean_audio = np.transpose(clean_audio)
                wind_noise = np.stack([wind_noise] * 2, axis=0)
            elif len(clean_audio) != 2:
                wind_noise = np.stack([wind_noise] * 2, axis=1)

        # 随机选择信噪比
        snr = random.uniform(*snr_wind_range)

        # 加噪
        noisy_audio = add_noise(clean_audio, wind_noise, snr)

        # 保存加噪后的音频
        sf.write(output_path, noisy_audio, sr)

    # 为交通噪声的干净语音加噪
    for i in range(traffic_noise_count):
        clean_index = traffic_indices[i]
        clean_audio_path = os.path.join(clean_audio_dir, clean_audio_files[clean_index])
        output_path = os.path.join(output_dir,
                                   f"{os.path.splitext(clean_audio_files[clean_index])[0]}_traffic_noisy.wav")
        traffic_noise_path = os.path.join(traffic_noisy_dir, traffic_noise_files[i])

        # 读取干净语音和交通噪声
        clean_audio, sr = librosa.load(clean_audio_path)
        traffic_noise, sr_noisy = librosa.load(traffic_noise_path)

        if sr_noisy != sr:
            traffic_noise = librosa.resample(traffic_noise, sr_noisy, sr)

        # 转换噪声为双通道（如果干净语音是双通道）
        if clean_audio.ndim == 2:
            if len(clean_audio) == 2:
                clean_audio = np.transpose(clean_audio)
                traffic_noise = np.stack([traffic_noise] * 2, axis=0)
            elif len(clean_audio) != 2:
                traffic_noise = np.stack([traffic_noise] * 2, axis=1)

        # 随机选择信噪比
        snr = random.uniform(*snr_traffic_range)

        # 加噪
        noisy_audio = add_noise(clean_audio, traffic_noise, snr)

        # 保存加噪后的音频
        sf.write(output_path, noisy_audio, sr)

#  数据集中有1068条干净语音与1068条加噪语音，但他们的label是一一对应的，该函数是为了复制一份label并改名为加噪后的格式以便于处理
def add_f0():
    folder_f0 = "/home/GaoYA/Code/DANet-main/data/f0ref_with_noisy"
    folder_cfp = "/home/GaoYA/Code/DANet-main/data/wav_with_noisy"

    # 遍历CFP特征文件夹中的所有文件（其中包含1068干净CFP+1068加噪CFP）
    for file_name_cfp in os.listdir(folder_cfp):
        file_path_cfp = os.path.join(folder_cfp, file_name_cfp)

        # 检查文件名是否以特定后缀结尾
        if file_name_cfp.endswith("_babble_noisy.wav"):

            # 提取文件名的前缀
            prefix = os.path.splitext(file_name_cfp)[0]
            prefix = prefix.replace("_babble_noisy", "")

            # 构建对应的在f0文件夹内的文件名
            txt_file_name = prefix + ".txt"

            # 检查文件夹A中是否存在对应的txt文件
            txt_file_path = os.path.join(folder_f0, txt_file_name)
            if os.path.isfile(txt_file_path):
                # 构建目标文件的路径
                target_file_path = os.path.join(folder_f0, prefix + "_babble_noisy.txt")
                shutil.copy(txt_file_path, target_file_path)

        if file_name_cfp.endswith("_traffic_noisy.wav"):

            # 提取文件名的前缀
            prefix = os.path.splitext(file_name_cfp)[0]
            prefix = prefix.replace("_traffic_noisy", "")

            # 构建对应的在f0文件夹内的文件名
            txt_file_name = prefix + ".txt"

            # 检查文件夹A中是否存在对应的txt文件
            txt_file_path = os.path.join(folder_f0, txt_file_name)
            if os.path.isfile(txt_file_path):
                # 构建目标文件的路径
                target_file_path = os.path.join(folder_f0, prefix + "_traffic_noisy.txt")
                shutil.copy(txt_file_path, target_file_path)

        if file_name_cfp.endswith("_wind_noisy.wav"):

            # 提取文件名的前缀
            prefix = os.path.splitext(file_name_cfp)[0]
            prefix = prefix.replace("_wind_noisy", "")

            # 构建对应的在f0文件夹内的文件名
            txt_file_name = prefix + ".txt"

            # 检查文件夹A中是否存在对应的txt文件
            txt_file_path = os.path.join(folder_f0, txt_file_name)
            if os.path.isfile(txt_file_path):
                # 构建目标文件的路径
                target_file_path = os.path.join(folder_f0, prefix + "_wind_noisy.txt")
                shutil.copy(txt_file_path, target_file_path)

# 根据wav和f0样本创建数据目录(.txt)
def creat_list():
    file_path = "/home/GaoYA/Code/DANet-main/add_noisy/output"
    output_path = "/home/GaoYA/Code/DANet-main/data/train_data_with_noisy.txt"
    with open(output_path, "w") as file:
        for filename in os.listdir(file_path):
            if filename.endswith(".wav"):
                filename = os.path.splitext(filename)[0] + ".npy"
                file.write(filename + "\n")

if __name__ == "__main__":
    add_single_add()
    main()
    count_duration()
    noisy_segment()
    add_f0()
    creat_list()

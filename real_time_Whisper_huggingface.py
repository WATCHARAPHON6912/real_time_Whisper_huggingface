import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import torch
from transformers import pipeline
import thaispellcheck

# การตั้งค่าพารามิเตอร์สำหรับการบันทึกเสียง
CHUNK = 1024 * 5
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
THRESHOLD = 30
SILENCE_DURATION = 2

# การตั้งค่าโมเดล Whisper สำหรับการถอดเสียง
MODEL_NAME = "whisper-th-medium-Xi"
device = 0 if torch.cuda.is_available() else "cpu"
print(device)
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

# ฟังก์ชันสำหรับการถอดเสียง
def pre(audio):
    return pipe(audio, batch_size=16, return_timestamps=False)["text"]

# ฟังก์ชันสำหรับการจับเสียงและแสดงกราฟแบบเรียลไทม์
def audio_capture(show_plot=False):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    if show_plot:
        # สร้างกราฟ
        plt.ion()
        fig, ax = plt.subplots()
        x = np.arange(0, 4 * CHUNK, 2)
        line, = ax.plot(x, np.random.rand(2 * CHUNK))
        ax.set_ylim(-3000, 3000)
        ax.set_xlim(0, 4 * CHUNK)
        plt.show()

    try:
        while True:
            silence_counter = 0
            detect = False
            frames = []
            while True:
                # อ่านข้อมูลจาก stream
                data = stream.read(CHUNK)
                np_data = np.frombuffer(data, dtype=np.int16)
                frames.append(data)

                if show_plot:
                    # อัปเดตกราฟแบบเรียลไทม์
                    data_plot = np.concatenate((line.get_ydata()[CHUNK:], np_data))
                    line.set_ydata(data_plot)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                # ตรวจสอบว่ามีการพูดหรือไม่
                if np.abs(np_data).mean() > THRESHOLD:
                    print("ตรวจพบการพูด...")
                    detect = True
                    silence_counter = 0
                else:
                    silence_counter += 1

                if silence_counter > int(SILENCE_DURATION * RATE / CHUNK) and detect:
                    print("กำลังประมวลผลเสียง")
                    yield b''.join(frames)
                    break
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# ใช้ฟังก์ชัน audio_capture และ pre สำหรับการจับเสียงและถอดเสียง
for frame in audio_capture(show_plot=True):  # เปลี่ยนเป็น False หากไม่ต้องการแสดงกราฟ
    audio_data = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
    print(thaispellcheck.check(pre(audio_data),autocorrect=True))


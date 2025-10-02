from faster_whisper import WhisperModel

model_size = "SeamlessX/malaysian-faster-whisper-large-v3-turbo-v3-ct2"

model = WhisperModel(model_size, device="cuda", compute_type="float16")

segments, info = model.transcribe("audio3.mp3")
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
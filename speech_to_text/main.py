import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    while True:
        print("\nSpeak..")
        audio_text = r.listen(source)
        print("Recognizing..")
        try:
            print("Text Recognized: " + r.recognize_google(audio_text) + '\n')
        except:
            print("[Error] Try again..")
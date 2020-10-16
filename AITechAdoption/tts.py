import requests
import http.client
import sys
import wave


def get_token(subscription_key):
    fetch_token_url = 'https://westus.api.cognitive.microsoft.com/sts/v1.0/issueToken'
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key
    }
    response = requests.post(fetch_token_url, headers=headers)
    access_token = str(response.text)
    print("\nAlready get access token")
    return access_token


 
def get_speech(access_token,text):
    headers = {"Content-type": "application/ssml+xml", 
			"X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
			"Authorization": "Bearer " + access_token, 
			"User-Agent": "TTSForPython"}
    body= '<speak version="1.0" xml:lang="en-US"><voice xml:lang="en-US" xml:gender="Female" name="en-US-AriaRUS">'+text+'</voice></speak>'			
    #Connect to server to synthesize the wave
    print ("\nConnect to server to synthesize the wave")
    conn = http.client.HTTPSConnection("westus.tts.speech.microsoft.com")
    conn.request("POST", "/cognitiveservices/v1", body, headers)
    response = conn.getresponse()
    print(response.status, response.reason)
    data = response.read()
    conn.close()
    print("The synthesized wave length: %d" %(len(data)))
    f = wave.open(r"output.wav", "wb")
    f.setnchannels(1)#单声道
    f.setframerate(24000)#采样率
    f.setsampwidth(2)#sample width 2 bytes(16 bits)
    f.writeframes(data)
    f.close()
    
    
if __name__ == "__main__":
    subscription_key = 'YOUR SUBSCRIPTION KEY'
    #subscription_key = '6081382052df493a9d5cd3123d09f12f'
    access_token = get_token(subscription_key)
    get_speech(access_token,sys.argv[1])

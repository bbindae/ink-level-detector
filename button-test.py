import RPi.GPIO as GPIO
import time

button = 12

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(button, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

while True:
	if GPIO.input(button) == GPIO.HIGH:
		print("Button was pushed!")
		time.sleep(1)
		

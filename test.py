import pywhatkit

message='hello'
number='+919562688725'
# Send a WhatsApp Message to the contact instantly (gives 10s to load web client before sending)
pywhatkit.sendwhatmsg_instantly(number, message, 10)


#The provided Python code uses the pywhatkit library to send a WhatsApp message instantly.
# The sendwhatmsg_instantly function is invoked with the recipient's phone number (number variable) and the message content (message variable).
# The third argument, 10, represents the wait time in seconds for the web client to load before sending the message.
# This delay allows the library to initialize the web interface for sending messages on WhatsApp. The message "hello" will be sent to the specified phone number once the web client is ready.

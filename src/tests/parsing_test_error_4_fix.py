message="hello world"

for x in range(len(message)):
        if (ord(message[x]))==32 or (ord(message[x])>=48 and ord(message[x])<=57):
            continue
        elif (ord(message[x])==39 or ord(message[x])==34):
            message=message.replace(message[x],"")
        message=message.replace(message[x],chr((ord(message[x]))+key))
        print(message)

from gptquery import FString


fstring = FString("{x} {y}")
print(fstring)
d = {"x": 1}
fstring = fstring.format(**d)
print(fstring)
fstring = fstring.format(y=2)
print(fstring)
fstring = str(fstring)
print(fstring)
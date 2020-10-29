import re

#Return a list containing every occurrence of "ai":

txt = "[('keyboard-typo', 0.2), ('natural-typo', 0.2), ('truncate', 0.2)]"
x = re.findall(r"\([0-9a-zA-Z,'\-\. ]*\)", txt)
y = re.findall(r"'.*'",x[0])[0][1:-1]
z = re.findall(r"[0-9\.]+",x[0])[0]
print(x)
print(y)
print(z)
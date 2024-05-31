import re

# Basic Matching
pattern = r"hello"
text = "Hello, World! Hello, Python!"
matches = re.findall(pattern, text, re.IGNORECASE)
print(matches)  # Output: ['Hello', 'Hello']

# Character Classes
pattern = r"[aeiou]"
text = "Hello, World!"
vowels = re.findall(pattern, text, re.IGNORECASE)
print(vowels)  # Output: ['e', 'o', 'o']

# Quantifiers
pattern = r"\d{2,4}"
text = "Year: 2023, Month: 08"
numbers = re.findall(pattern, text)
print(numbers)  # Output: ['2023', '08']

# Anchors
pattern = r"^Hello"
text = "Hello, World! Hello, Python!"
matches = re.findall(pattern, text)
print(matches)  # Output: ['Hello']

# Grouping and Capturing
pattern = r"(H\w+) (\w+)"
text = "Hello World, Hello Python"
matches = re.findall(pattern, text)
print(matches)  # Output: [('Hello', 'World'), ('Hello', 'Python')]

# Alternation
pattern = r"Hello|Python"
text = "Hello, Python!"
matches = re.findall(pattern, text)
print(matches)  # Output: ['Hello', 'Python']

# Lookahead
pattern = r"\d(?= dollars)"
text = "5 dollars, 10 dollars"
matches = re.findall(pattern, text)
print(matches)  # Output: ['5']

# Substitution
pattern = r"\bapple\b"
text = "An apple is not an orange."
new_text = re.sub(pattern, "banana", text, flags=re.IGNORECASE)
print(new_text)  # Output: "An banana is not an orange."

# Flags and Modifiers
pattern = r"hello"
text = "Hello, World! Hello, Python!"
matches = re.findall(pattern, text, re.IGNORECASE)
print(matches)  # Output: ['Hello', 'Hello']


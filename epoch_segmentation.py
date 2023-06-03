Boredom = [1, 2, 3, 5, 7, 9, 15, 17, 26, 27, 28, 29, 32, 33, 38, 40, 49, 50, 51, 53, 57, 61, 68, 72, 76, 77, 79, 82, 85, 86, 88, 90, 94, 96, 97, 98, 99, 100, 102, 106, 110, 113, 114, 117, 118, 120]
Anger = [30, 69, 70, 87]
Joy = [4, 12, 14, 18, 19, 20, 21, 23, 25, 35, 37, 39, 42, 43, 46, 48, 54, 59, 66, 67, 74, 80, 93, 105, 111, 116]
Admiration = [13, 22, 58, 63, 64, 119]
Arousal = [6, 11, 36, 45, 60, 75, 83, 84, 89, 91, 95, 109]
Disgust = [31, 41, 52, 71, 73, 81, 101, 112, 115]
Sadness = [8, 10, 16, 24, 44, 47, 55, 56, 65, 92, 103, 104]
Anxiety = [34, 62, 78, 107, 108]

Positive = []
Positive.extend(Joy)
Positive.extend(Admiration)
Positive.extend(Arousal)
Negative = []
Negative.extend(Anger)
Negative.extend(Disgust)
Negative.extend(Sadness)
Negative.extend(Anxiety)
Neutral = []
Neutral.extend(Boredom)

all=[]
all.extend(Positive)
all.extend(Neutral)
all.extend(Negative)
print(len(all))
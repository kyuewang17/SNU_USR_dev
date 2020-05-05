from PIL import Image

width = 300 # Expected Width of generated Image
height = 100 # Height of generated Image

specratio = 255*6 / width

print ("SpecRatio: " + str(specratio))

red = 255
green = 0
blue = 0

colors = []

step = int(round(specratio))

for u in range (0, height):
    for i in range (0, 255*6+1, step):
        if i > 0 and i <= 255:
            blue += step
        elif i > 255 and i <= 255*2:
            red -= step
        elif i > 255*2 and i <= 255*3:
            green += step
        elif i > 255*3 and i <= 255*4:
            blue -= step
        elif i > 255*4 and i <= 255*5:
            red += step
        elif i > 255*5 and i <= 255*6:
            green -= step

        colors.append((red, green, blue))

newwidth = int(i/step+1) # Generated Width of Image without producing Float-Numbers

print (str(colors))
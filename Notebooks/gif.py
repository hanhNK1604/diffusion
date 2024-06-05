import imageio 

path = [
    r"C:/Users/Admin/Downloads\image0.png",
    r"C:\Users\Admin\Downloads\image1.png",
    r"C:\Users\Admin\Downloads\image2.png",
    r"C:\Users\Admin\Downloads\image3.png",
    r"C:\Users\Admin\Downloads\image4.png",
    r"C:\Users\Admin\Downloads\image5.png",
    r"C:\Users\Admin\Downloads\image6.png",
    r"C:\Users\Admin\Downloads\image7.png",
    r"C:\Users\Admin\Downloads\image8.png",
    r"C:\Users\Admin\Downloads\image9.png",
    r"C:\Users\Admin\Downloads\image10.png",
]

image = [] 
for filename in path: 
    image.append(imageio.imread(filename))

imageio.mimsave("result.gif", image, 'GIF', duration=1.5, loop=0)
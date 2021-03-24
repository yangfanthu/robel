import robel
import gym
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import cv2


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image

if __name__ == "__main__":
    x = np.array([0,1,2])
    y = np.array([1,0,1])
    # dots = np.array([[0,1],[1,0],[2,1]])
    figure1 = plt.figure()
    plt.xlim([0,5])
    plt.ylim([0,5])
    plt.plot(x,y)
    img1 = fig2data(figure1)
    img1 = Image.fromarray(img1)
    x = np.array([0,1,2])
    y = np.array([0,1,0])
    figure2 = plt.figure()
    plt.plot(x,y)
    img2 = fig2data(figure2)
    img2 = Image.fromarray(img2)
    final_img = Image.new('RGB',(640, 960), color="white")
    final_img.paste(img1, (0,0))
    final_img.paste(img2, (0,480))
    final_img.show()
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # array = np.array(img)




    # env = gym.make("DClawTurnFixedD2-v0")
    # obs = env.reset()
    # done = False
    # episodes = 10 
    # for i in range(episodes):
    #     while not done:
    #         env.render()
    #         action = env.action_space.sample()
    #         obs, reward, done, info = env.step(action)
    #     obs = env.reset()
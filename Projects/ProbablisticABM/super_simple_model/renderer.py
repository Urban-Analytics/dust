import imageio
import time
import os


def render_agent():
    files = sorted(os.listdir('output'))
    print('{} frames generated.'.format(len(files)))
    images = []
    for filename in files:
        images.append(imageio.imread('output/{}'.format(filename)))
    imageio.mimsave('outputGIFs/{}.mp4'.format(time.time()), images)


def clear_output_folder():
    folder = 'output'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
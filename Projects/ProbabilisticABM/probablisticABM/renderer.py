import imageio
import time
import os


def render_agent():
    files = sorted(os.listdir('output'))
    print('Renderer: {} frame(s) found...'.format(len(files)))
    images = []
    i = 1
    for filename in files:
        images.append(imageio.imread('output/{}'.format(filename)))
        print('Rendering frame {} of {}'.format(i, len(files)))
        i += 1
    imageio.mimsave('video_output/{}.mp4'.format(time.time()), images)


def clear_output_folder():
    folder = 'output'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

# Makefile for calibration process

all: inputs/activation.dat all_frames outputs/figures/speed outputs/figures/activation_rate outputs/data

inputs/activation.dat: ../grand_central_terminal_data/GCT_final_real_data/activation.dat
	cp ../grand_central_terminal_data/GCT_final_real_data/activation.dat inputs/activation.dat

inputs/frames.tar.gz: ../grand_central_terminal_data/GCT_final_real_data/frames.tar.gz
	cp ../grand_central_terminal_data/GCT_final_real_data/frames.tar.gz inputs/frames.tar.gz

inputs/frames:
	cd inputs/ && mkdir frames && touch frames/README.md

all_frames: inputs/frames inputs/frames.tar.gz
	cd inputs/ && tar -xzf frames.tar.gz -C frames/

zip_outputs:
	tar -czf all_outputs.tar.gz outputs/

unzip_outputs:
	tar -xzf all_outputs.tar.gz outputs/

outputs/figures:
	cd outputs/ && mkdir figures

outputs/figures/speed: outputs/figures
	cd outputs/figures/ && mkdir speed

outputs/figures/activation_rate: outputs/figures
	cd outputs/figures/ && mkdir activation_rate

outputs/data:
	cd outputs/ && mkdir data
